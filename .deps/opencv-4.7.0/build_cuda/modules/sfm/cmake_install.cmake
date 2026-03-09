# Install script for directory: /media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/media/yunxiangyang/Train/omniGS/.deps/opencv-install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibsx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so.4.7.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so.407"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/media/yunxiangyang/Train/omniGS/.deps/opencv-install/lib:/home/yunxiangyang/miniconda3/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY OPTIONAL FILES
    "/media/yunxiangyang/Train/omniGS/.deps/opencv-4.7.0/build_cuda/lib/libopencv_sfm.so.4.7.0"
    "/media/yunxiangyang/Train/omniGS/.deps/opencv-4.7.0/build_cuda/lib/libopencv_sfm.so.407"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so.4.7.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so.407"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/media/yunxiangyang/Train/omniGS/.deps/opencv-4.7.0/build_cuda/lib:/home/yunxiangyang/miniconda3/lib:"
           NEW_RPATH "/media/yunxiangyang/Train/omniGS/.deps/opencv-install/lib:/home/yunxiangyang/miniconda3/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so"
         RPATH "/media/yunxiangyang/Train/omniGS/.deps/opencv-install/lib:/home/yunxiangyang/miniconda3/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv-4.7.0/build_cuda/lib/libopencv_sfm.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so"
         OLD_RPATH "/media/yunxiangyang/Train/omniGS/.deps/opencv-4.7.0/build_cuda/lib:/home/yunxiangyang/miniconda3/lib:"
         NEW_RPATH "/media/yunxiangyang/Train/omniGS/.deps/opencv-install/lib:/home/yunxiangyang/miniconda3/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_sfm.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm/include/opencv2/sfm.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/sfm" TYPE FILE OPTIONAL FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm/include/opencv2/sfm/conditioning.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/sfm" TYPE FILE OPTIONAL FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm/include/opencv2/sfm/fundamental.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/sfm" TYPE FILE OPTIONAL FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm/include/opencv2/sfm/io.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/sfm" TYPE FILE OPTIONAL FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm/include/opencv2/sfm/numeric.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/sfm" TYPE FILE OPTIONAL FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm/include/opencv2/sfm/projection.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/sfm" TYPE FILE OPTIONAL FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm/include/opencv2/sfm/reconstruct.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/sfm" TYPE FILE OPTIONAL FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm/include/opencv2/sfm/robust.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/sfm" TYPE FILE OPTIONAL FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm/include/opencv2/sfm/simple_pipeline.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/sfm" TYPE FILE OPTIONAL FILES "/media/yunxiangyang/Train/omniGS/.deps/opencv_contrib-4.7.0/modules/sfm/include/opencv2/sfm/triangulation.hpp")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/media/yunxiangyang/Train/omniGS/.deps/opencv-4.7.0/build_cuda/modules/sfm/src/libmv/cmake_install.cmake")

endif()

