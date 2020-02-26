/*
 * TI Voxel SDK example.
 *
 * Copyright (c) 2014 Texas Instruments Inc.
 */
//./write2hardware -v 451 -p 9105 -s 12016791199903 -f none -t depth


#include "CameraSystem.h"

#include "SimpleOpt.h"
#include "Common.h"
#include "UVCStreamer.h"
#include <iomanip>
#include <fstream>

using namespace Voxel;

enum Options
{
  VENDOR_ID = 0,
  PRODUCT_ID = 1,
  SERIAL_NUMBER = 2,
  DUMP_FILE = 3,
  NUM_OF_FRAMES = 4,
  CAPTURE_TYPE = 5
};

Vector<CSimpleOpt::SOption> argumentSpecifications = 
{
  { VENDOR_ID,    "-v", SO_REQ_SEP, "Vendor ID of the USB device (hexadecimal)"}, // Only worker count is needed here
  { PRODUCT_ID,   "-p", SO_REQ_SEP, "Comma separated list of Product IDs of the USB devices (hexadecimal)"},
  { SERIAL_NUMBER,"-s", SO_REQ_SEP, "Serial number of the USB device (string)"},
  { DUMP_FILE,    "-f", SO_REQ_SEP, "Name of the file to dump extracted frames"},
  { NUM_OF_FRAMES,"-n", SO_REQ_SEP, "Number of frames to dump [default = 1]"},
  { CAPTURE_TYPE, "-t", SO_REQ_SEP, "Type of capture (raw/raw_processed/depth/pointcloud) [default = raw]" },
  SO_END_OF_OPTIONS
};

void help()
{
  std::cerr << "DepthCapture v1.0" << std::endl;
  
  CSimpleOpt::SOption *option = argumentSpecifications.data();
  
  while(option->nId >= 0)
  {
    std::cerr << option->pszArg << " " << option->helpInfo << std::endl;
    option++;
  }
}


int main(int argc, char *argv[])
{
  std::stringstream is;

  CSimpleOpt s(argc, argv, argumentSpecifications);
  
  logger.setDefaultLogLevel(LOG_INFO);
  
  uint16_t vid = 0;
  
  Vector<uint16_t> pids;
  String serialNumber;
  String dumpFileName;

  String type = "raw";
  
  int32_t frameCount = 1;
  
  char *endptr;
  
  while (s.Next())
  {
    if (s.LastError() != SO_SUCCESS)
    {
      std::cerr << s.GetLastErrorText(s.LastError()) << ": '" << s.OptionText() << "' (use -h to get command line help)" << std::endl;
      help();
      return -1;
    }
    
    Vector<String> splits;
    switch (s.OptionId())
    {
      case VENDOR_ID:
        vid = (uint16_t)strtol(s.OptionArg(), &endptr, 16);
        break;
        
      case PRODUCT_ID:
        split(s.OptionArg(), ',', splits);
        
        for(auto &s1: splits)
          pids.push_back((uint16_t)strtol(s1.c_str(), &endptr, 16));
        
        break;
        
      case SERIAL_NUMBER:
        serialNumber = s.OptionArg();
        break;
        
      case DUMP_FILE:
        dumpFileName = s.OptionArg();
        break;
        
      case NUM_OF_FRAMES:
        frameCount = (int32_t)strtol(s.OptionArg(), &endptr, 10);
        break;

      case CAPTURE_TYPE:
        type = s.OptionArg();
        break;
        
      default:
        help();
        break;
    };
  }
  
  if(vid == 0 || pids.size() == 0 || pids[0] == 0 || dumpFileName.size() == 0)
  {
    std::cerr << "Required argument missing." << std::endl;
    help();
    return -1;
  }

  if (type != "raw" && type != "raw_processed" && type != "depth" && type != "pointcloud")
  {
    std::cerr << "Unknown type '" << type << "'" << std::endl;
    help();
    return -1;
  }
  
  CameraSystem sys;
  
  // Get all valid detected devices
  const Vector<DevicePtr> &devices = sys.scan();
  
  DevicePtr toConnect;
  
  std::cerr << "Detected devices: " << std::endl;
  for(auto &d: devices)
  {
    std::cerr << d->id() << std::endl;
    
    if(d->interfaceID() == Device::USB)
    {
      USBDevice &usb = (USBDevice &)*d;
      
      if(usb.vendorID() == vid && (serialNumber.size() == 0 || usb.serialNumber() == serialNumber))
      {
        for(auto pid: pids)
          if(usb.productID() == pid)
            toConnect = d;
      }
    }
  }
  
  if(!toConnect)
  {
    std::cerr << "No valid device found for the specified VID:PID:serialnumber" << std::endl;
    return -1;
  }
    
  DepthCameraPtr depthCamera = sys.connect(toConnect);

  bool read, write;
  read = depthCamera->configFile.read("TintinCDKCamera.conf");
  // read = depthCamera->configFile.read("RTS5825Camera.conf");
  if(read)
    std::cout<< "Configuration file loaded sucessufully" << std::endl;
  else
    std::cout<< "Configuration file could not be loaded" << std::endl;

  write = depthCamera->configFile.writeToHardware();
  if(write)
    std::cout<< "Main configuration file written sucessufully" << std::endl;
  else
    std::cout<< "Main configuration file could not be written" << std::endl;
  
  
  std::vector<int> ids{102};
  for(auto id=ids.begin(); id!=ids.end(); id++){
    write = depthCamera->configFile.saveCameraProfileToHardware(*id);

    if(write)
      std::cout<< "Profile written sucessufully" << std::endl;
    else
      std::cout<< "Profile could not be written" << std::endl;

  }

  
  return 0;
}