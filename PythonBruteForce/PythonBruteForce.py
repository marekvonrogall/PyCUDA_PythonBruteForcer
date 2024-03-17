import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

def get_password():
    return input("Please enter your password: ")

def get_thread_count(characters, password_length):
    print("(info) Max combinations for this password with " + str(len(characters)) + " possible characters are:", get_combinations(password_length, characters))
    return int(input("Please enter the number of threads: "))

def get_combinations(password_length, characters):
    return len(characters) ** password_length

def get_portion_size(combinations, thread_count):
    return combinations / thread_count

def get_starting_strings(portion_size, thread_count, password_length, characters):
    print("Calculating entry points for " + str(thread_count) + " threads... (This might take a while)")
    starting_strings = []
    big_int_value = len(characters)  # Precompute length of characters string

    for i in range(thread_count):
        current_portion = portion_size * i
        start_string = []
        for _ in range(password_length):
            index = int(current_portion) % big_int_value
            start_string.append(characters[index])
            current_portion /= big_int_value
        starting_strings.append(''.join(start_string[::-1]))  # Reverse and append to starting_strings
        
    print("Calculating entry points complete!")
    print("------------------------------------------------------------")
    return starting_strings

# CUDA kernel for searching passwords
cuda_kernel = """
__global__ void search_password(const char *user_password, char *starting_strings, int num_threads, int user_password_length, int update_rate, volatile int *password_found_flag, volatile long long int *attempts) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_threads) {
        // Assuming maximum password length of 16
        char password[128];

        // Copy starting string for this thread
        for (int i = 0; i < user_password_length; ++i) {
            password[i] = starting_strings[idx * user_password_length + i];
        }

        int counter = 0;

        while (true) {
            // Check if password found by another thread
            if (*password_found_flag) {
                break;
            }

            // Compare passwords
            bool found = true;
            for (int i = 0; i < user_password_length; ++i) {
                if (password[i] != user_password[i]) {
                    found = false;
                    break;
                }
            }

            if (found) {
                printf("Thread %d: Password found: %s in %d attempts.\\n", idx, password, counter);
                atomicExch((int*)password_found_flag, 1);
                break;
            }

            // Increment the password
            for (int i = user_password_length - 1; i >= 0; --i) {
                if (password[i] == 'z') {
                    password[i] = 'a';  // wrap around from 'z' to 'a'
                } else {
                    ++password[i];
                    break;
                }
            }
            counter++;

            if (update_rate > 0 && counter % update_rate == 0) {
                printf("Thread %d: A: %d P: %s\\n", idx, counter, password);
            }
        }

        // Store attempts made by this thread
        attempts[idx] = counter;
    }
}
"""

def local_cuda_search(user_password, starting_strings, password_length, update_rate, block_size):
    num_threads = len(starting_strings)
    password_found_flag = np.zeros(1, dtype=np.int32)
    attempts = np.zeros(num_threads, dtype=np.int64)  # Array to store attempts made by each thread

    # Encode starting strings to bytes and concatenate
    starting_strings_bytes = b''.join(s.encode('utf-8') for s in starting_strings).ljust(num_threads * password_length, b'\0')

    # Allocate memory on GPU for starting strings
    starting_strings_gpu = cuda.mem_alloc(num_threads * password_length)
    cuda.memcpy_htod(starting_strings_gpu, starting_strings_bytes)

    # Allocate memory on GPU for user_password
    user_password_gpu = cuda.mem_alloc(password_length)
    cuda.memcpy_htod(user_password_gpu, user_password.encode('utf-8'))

    # Allocate memory on GPU for attempts
    attempts_gpu = cuda.mem_alloc(num_threads * np.dtype(np.int64).itemsize)

    # Compile CUDA kernel
    mod = SourceModule(cuda_kernel)

    # Get function handle
    search_func = mod.get_function("search_password")

    # Calculate the number of blocks needed
    num_blocks = (num_threads + block_size - 1) // block_size

    # Launch kernel with multiple blocks
    search_func(user_password_gpu, starting_strings_gpu, np.int32(num_threads), np.int32(password_length), np.int32(update_rate), cuda.InOut(password_found_flag), attempts_gpu, block=(block_size, 1, 1), grid=(num_blocks, 1))

    # Copy attempts from GPU to CPU
    cuda.memcpy_dtoh(attempts, attempts_gpu)

    # Wait for kernel to finish
    cuda.Context.synchronize()

    # Calculate total attempts made by all threads
    total_attempts = np.sum(attempts)
    print("Total attempts made by all threads:", total_attempts)

if __name__ == "__main__":
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    while True:
        print("------------------------------------------------------------")
        print("(info) Current character set is:", characters)
        user_password = get_password()
        password_length = len(user_password)
        
        thread_count = get_thread_count(characters, password_length)
        portion_size = get_portion_size(get_combinations(password_length, characters), thread_count)
    
        print("------------------------------------------------------------")
        print("Password:", user_password)
        print("Password Length:", password_length)
        print("Thread Count:", thread_count)
        print("Portion Size:", portion_size)
        print("------------------------------------------------------------")

        starting_strings = get_starting_strings(portion_size, thread_count, password_length, characters)
        
        # Parameters for CUDA search
        update_rate = 0  # How often the current passwort each thread is trying gets printed. 0 = no updates
        threads_per_block = 1024  # Maximum threads per block supported by your device

        local_cuda_search(user_password, starting_strings, password_length, update_rate, threads_per_block)
