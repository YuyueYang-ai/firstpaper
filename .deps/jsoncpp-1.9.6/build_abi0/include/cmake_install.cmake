# Install script for directory: /media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-install")
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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/json" TYPE FILE FILES
    "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include/json/allocator.h"
    "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include/json/assertions.h"
    "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include/json/config.h"
    "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include/json/forwards.h"
    "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include/json/json.h"
    "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include/json/json_features.h"
    "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include/json/reader.h"
    "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include/json/value.h"
    "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include/json/version.h"
    "/media/yunxiangyang/Train/omniGS/.deps/jsoncpp-1.9.6/include/json/writer.h"
    )
endif()

