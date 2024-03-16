import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

def get_password():
    password = input("Please enter your password: ")
    return password

def get_thread_count(characters, password_length):
    print("Threads needed so every thread checks...")
    print("1 password: " + str(get_max_combinations(characters, password_length, 1)))
    print("10000000 passwords: " + str(get_max_combinations(characters, password_length, 1000000)))
    processor_count = input("Please enter the number of threads: ")
    return int(processor_count)

def get_combinations(password_length, characters):
    combinations = len(characters) ** password_length
    return combinations

def get_portion_size(combinations, processor_count):
    return combinations / processor_count

def get_starting_strings(portion_size, processor_count, password_length, characters):
    print("Calculating starting strings... (This could take a while)")
    starting_strings = []
    for i in range(processor_count):
        current_portion = portion_size * i
        start_string = ""
        for _ in range(password_length):
            big_int_value = len(characters)
            start_string += characters[int(current_portion) % int(big_int_value)]
            current_portion /= big_int_value
        start_string = ''.join(reversed(start_string))
        starting_strings.append(start_string)
    return starting_strings

def get_max_combinations(characters, password_length, passwords_per_thread):
    combinations = len(characters) ** password_length
    return combinations / passwords_per_thread

# CUDA kernel for searching passwords
cuda_kernel = """
__global__ void search_password(const char *user_password, char *starting_strings, int num_threads, int password_length, int update_rate, volatile int *password_found_flag) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_threads) {
        char password[128];  // assuming max password length of 16

        for (int i = 0; i < password_length; ++i) {
            password[i] = starting_strings[idx * password_length + i];
        }

        //printf("Thread %d starting with: %s\\n", idx, password);

        int counter = 0;

        while (true) {
            // Check if password found by another thread
            if (*password_found_flag) {
                break;
            }

            // Compare passwords
            bool found = true;
            for (int i = 0; i < password_length; ++i) {
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
            for (int i = password_length - 1; i >= 0; --i) {
                if (password[i] == 'z') {
                    password[i] = 'a';  // wrap around from 'z' to 'a'
                } else {
                    ++password[i];
                    break;
                }
            }
            counter++;

            /*if (counter % update_rate == 0) {
                printf("Thread %d: A: %d P: %s\\n", idx, counter, password);
            }*/
        }
    }
}
"""

def local_cuda_search(user_password, starting_strings, password_length, update_rate, block_size):
    num_threads = len(starting_strings)
    password_found_flag = np.zeros(1, dtype=np.int32)

    # Encode starting strings to bytes and concatenate
    starting_strings_bytes = b''.join(s.encode('utf-8') for s in starting_strings).ljust(num_threads * password_length, b'\0')

    # Allocate memory on GPU for starting strings
    starting_strings_gpu = cuda.mem_alloc(num_threads * password_length)
    cuda.memcpy_htod(starting_strings_gpu, starting_strings_bytes)

    # Allocate memory on GPU for user_password
    user_password_gpu = cuda.mem_alloc(password_length)
    cuda.memcpy_htod(user_password_gpu, user_password.encode('utf-8'))

    # Compile CUDA kernel
    mod = SourceModule(cuda_kernel)

    # Get function handle
    search_func = mod.get_function("search_password")

    # Calculate the number of blocks needed
    num_blocks = (num_threads + block_size - 1) // block_size

    # Launch kernel with multiple blocks
    search_func(user_password_gpu, starting_strings_gpu, np.int32(num_threads), np.int32(password_length), np.int32(update_rate), cuda.InOut(password_found_flag), block=(block_size, 1, 1), grid=(num_blocks, 1))

    # Wait for kernel to finish
    cuda.Context.synchronize()

if __name__ == "__main__":
    while True:
        user_password = get_password()
        password_length = len(user_password)
        characters = "abcdefghijklmnopqrstuvwxyz"
        
        thread_count = get_thread_count(characters, password_length)
        portion_size = get_portion_size(get_combinations(password_length, characters), thread_count)
    
        print("Password:", user_password)
        print("Password Length:", password_length)
        print("Thread Count:", thread_count)
        print("Portion Size:", portion_size)

        starting_strings = get_starting_strings(portion_size, thread_count, password_length, characters)
        #print(starting_strings)
        print("Calculating starting strings finished!")
        
        # Parameters for CUDA search
        update_rate = 100000000  # Define your update rate here
        threads_per_block = 1024  # Maximum threads per block supported by your device

        local_cuda_search(user_password, starting_strings, password_length, update_rate, threads_per_block)
