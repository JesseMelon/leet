cmake_minimum_required(VERSION 4.2)

project(Leet C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_C_COMPILER "gcc")

if(${DEBUG})
    add_compile_options(-g)
else()
    add_compile_options(-O2)
endif()

add_compile_options(-fsanitize=address -mno-avx512f)
add_link_options(-fsanitize=address)

# uthash sucks
# include_directories(path_to_uthash/include)

add_executable(leet_out ${INPUT})
