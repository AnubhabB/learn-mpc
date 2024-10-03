cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
# message (STATUS "CMAKE_C: ${CMAKE_C_COMPILER}")
# message (STATUS "CMAKE_CXX: ${CMAKE_CXX_COMPILER}")
# message (STATUS ${CMAKE_CXX_COMPILER_AR})
# message (STATUS ${CMAKE_CXX_COMPILER_RANLIB})
project(learn_mpc)

# Library definition
add_library(METAL_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/metaldef.cpp
        )

# Metal cpp headers
target_include_directories(METAL_CPP PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp"
        # "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp-extensions"
        )

# Metal cpp library (linker)
target_link_libraries(METAL_CPP
        "-framework Metal"
        # "-framework MetalKit"
        # "-framework AppKit"
        "-framework Foundation"
        # "-framework QuartzCore"
        )

option(EXAMPLE "Example to run" "1-vector-add")

IF(IS_DIRECTORY ${EXAMPLE})
    # Get project name and all sources
    get_filename_component(project_name ${EXAMPLE} NAME)
    FILE(GLOB SRC ${EXAMPLE}/metal.cpp)

    message(STATUS "File: ${SRC}")
    
    # Create executable and link target
    add_executable(${project_name}-out ${SRC})
    message(STATUS "Linking ${METAL_CPP}")
    target_link_libraries(${project_name}-out METAL_CPP)
    # Find the JPEG package
    find_package(JPEG REQUIRED)
    # If find_package doesn't work, you can set paths manually:
    if(NOT JPEG_FOUND)
        set(JPEG_INCLUDE_DIR "/usr/local/opt/jpeg/include")
        set(JPEG_LIBRARY "/usr/local/opt/jpeg/lib/libjpeg.dylib")
        # For Apple Silicon Macs, you might need to use:
        # set(JPEG_INCLUDE_DIR "/opt/homebrew/opt/jpeg/include")
        # set(JPEG_LIBRARY "/opt/homebrew/opt/jpeg/lib/libjpeg.dylib")
    endif()
    
    # Add the include directories
    include_directories(${JPEG_INCLUDE_DIR})
    target_link_libraries(${project_name}-out ${JPEG_LIBRARY})

    message(STATUS "Adding ${project_name}")
ENDIF()