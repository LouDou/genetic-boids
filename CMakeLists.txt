cmake_minimum_required(VERSION 3.0)
project(genetic-boids)

SET(CMAKE_CXX_FLAGS "-std=c++20")

OPTION(FEATURE_RENDER_STATS "Enable support for rendering stats")
OPTION(FEATURE_RENDER_VIDEO "Enable support for rendering to video")
OPTION(FEATURE_CLI_OPTIONS "Enable support CLI options")

# Boids

add_executable(boids
    src/agent.cpp
    src/conditions.cpp
    src/neuralagent.cpp
    src/neuron.cpp
    src/random.cpp
    src/ui.cpp
    src/video.cpp
    src/main.cpp
)

if( ${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")

    set(USE_FLAGS "-s USE_SDL=2 -s USE_SDL_GFX=2 -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${USE_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${USE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${USE_FLAGS} -s ASYNCIFY")
    set(CMAKE_EXECUTABLE_SUFFIX .html)

else()

    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

    find_package(SDL2 CONFIG REQUIRED)
    target_link_libraries(boids PRIVATE SDL2::SDL2main SDL2::SDL2-static)

    find_package(sdl2-gfx CONFIG REQUIRED)
    target_link_libraries(boids PRIVATE SDL2::SDL2_gfx)

    if (FEATURE_RENDER_STATS)
        add_definitions(-DFEATURE_RENDER_STATS)
        target_link_libraries(boids PRIVATE fontconfig)
        find_package(SDL2_ttf CONFIG REQUIRED)
        target_link_libraries(boids PRIVATE $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>)
    endif() # FEATURE_RENDER_STATS

    if (FEATURE_RENDER_VIDEO)
        add_definitions(-DFEATURE_RENDER_VIDEO)
        find_package(FFMPEG REQUIRED)
        target_include_directories(boids PRIVATE ${FFMPEG_INCLUDE_DIRS})
        target_link_directories(boids PRIVATE ${FFMPEG_LIBRARY_DIRS})
        target_link_libraries(boids PRIVATE ${FFMPEG_LIBRARIES})
    endif() # FEATURE_RENDER_VIDEO

    if (FEATURE_CLI_OPTIONS)
        add_definitions(-DFEATURE_CLI_OPTIONS)
        find_package(argparse CONFIG REQUIRED)
        target_link_libraries(boids PRIVATE argparse::argparse)
    endif() # FEATURE_CLI_OPTIONS

endif() # Emscripten
