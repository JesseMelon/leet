cmake_minimum_required(VERSION 4.2)

project(Leet C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_C_COMPILER "gcc")

add_compile_options(-O2 -fsanitize=address)
add_link_options(-fsanitize=address)

# uthash sucks
# include_directories(path_to_uthash/include)

add_executable(leet_out ${INPUT})
