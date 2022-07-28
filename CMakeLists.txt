cmake_minimum_required(VERSION 3.0)
project(genetic-boids)

SET(CMAKE_CXX_FLAGS "-std=c++17 -fopenmp")

# Boids

add_executable(boids
    src/agent.cpp
    src/conditions.cpp
    src/neuralagent.cpp
    src/random.cpp
    src/main.cpp
)

find_package(SDL2 CONFIG REQUIRED)
target_link_libraries(boids PRIVATE SDL2::SDL2main SDL2::SDL2-static)

target_link_libraries(boids PRIVATE fontconfig)

find_package(sdl2-ttf CONFIG REQUIRED)
target_link_libraries(boids PRIVATE SDL2::SDL2_ttf)

find_package(sdl2-gfx CONFIG REQUIRED)
target_link_libraries(boids PRIVATE SDL2::SDL2_gfx)

find_package(nanoflann CONFIG REQUIRED)
target_link_libraries(boids PRIVATE nanoflann::nanoflann)
