cmake_minimum_required(VERSION 2.6)
project(chaos-spiral-defect)


## USER AREA
# Options are:
# - -DALLOW_INTERACTION 	Allow the user to interact with the simulation
# - -DTRACK_VALUES			Displays in a terminal some real time values
# - -DTIMER					Enables a timer to check the frame times and network times
add_definitions(-DALLOW_INTERACTION -DTRACK_VALUES -DTIMER)



## MAKEFILE CONSTRUCTION AREA
if(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES debug)
	set(CMAKE_BUILD_TYPE Debug)
else(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES debug)
	set(CMAKE_BUILD_TYPE Release)
endif(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES debug)

message("\n")
message("Build type is ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -pedantic -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
	message("Flags are: ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
	message("Flags are: ${CMAKE_CXX_FLAGS_RELEASE}")
endif(CMAKE_BUILD_TYPE MATCHES "Debug")

message("\n")

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(SDL2_ttf REQUIRED)


include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIRS} ${SDL2_TTF_INCLUDE_DIRS})

add_executable(visual-program
	src/main.cpp
	src/visual.cpp
	src/coloring.cpp
	src/window.cpp
	src/network/datafile.cpp
	src/network/serverclient.cpp
)

target_link_libraries(visual-program ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARIES} ${SDL2_TTF_LIBRARIES})
