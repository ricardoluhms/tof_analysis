/*
 * TI Voxel SDK example.
 *
 * Copyright (c) 2014 Texas Instruments Inc.
 */
//./DepthCapture -v 451 -p 9105 -s 12016791199903 -f none -t depth


#include "CameraSystem.h"

#include "SimpleOpt.h"
#include "Common.h"
#include "UVCStreamer.h"
#include <iomanip>
#include <fstream>

#include <map>
#include <bitset>

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

  /*
  Configuration c;
  if(c.getFirmwareFile(file))
    return true;
  // Try name as is, to see whether it is valid by itself. That is, 
  // it could be an absolute path or path relative to current working directory
  std::ifstream f(file, std::ios::binary);
  if(f.good())
    return true;
  return false;
  */

  // if(!depthCamera->setCameraProfile(102, true))
  //   std::cerr << "Could not set profile" << std::endl;

  std::cerr<<"==================================================\n";
  std::cerr << "Current profile id:" << depthCamera->getCurrentCameraProfileID() << std::endl;

  
  // Map<String, ParameterPtr> params= depthCamera->getParameters();
  // for(auto itr = params.begin(); itr != params.end(); ++itr) { 
  //   uint16_t slaveAddress, registerAddress;
  //   slaveAddress = (itr->second->address() & 0xFF00) >> 8;
  //   registerAddress = (itr->second->address() & 0xFF);
  //   std::cout << itr->first << '\n' << std::hex << itr->second->address()<< '\t' << std::hex << slaveAddress << '\t' << std::hex << registerAddress << "\n\n";
  // }
  // return 0;


  uint32_t address{0x5c01}, firmware{0};
  uint8_t major{0}, minor{0}, device{0};
  Ptr<RegisterProgrammer> programmer = depthCamera->getProgrammer();
  programmer->readRegister(address, firmware);
  minor = (firmware & 0b000011111);
  major = (firmware & 0b011100000)>>5;
  device =(firmware & 0b100000000)>>8;
  std::cerr<<"firmware: "<< (int)major<<"."<<(int)minor <<std::endl;
  std::cerr<<"device: "<< (int)device <<std::endl;
  std::cerr<<"--------------------------------------------------\n";

  // address =0x5c4c;
  // uint32_t value2get{0}, value2set{0b101000000}, mask{0b11111111111111111111111000111111};
  // if(programmer->readRegister(address, value2get))
  //   std::cout<< "old value register 0x4c: " << std::bitset<24>(value2get) <<"\n";
  // else
  //   std::cout<< "old value register 0x4c: could not read" <<"\n";
  // value2get &= mask;
  // value2set += value2get;
  // if(programmer->writeRegister(address, value2set))
  //   std::cout<< "new value register 0x4c: " << std::bitset<24>(value2set) <<"\n";
  // else
  //   std::cout<< "new value register 0x4c: could not set" <<"\n";

  // uint setvalue{66666};
  // depthCamera->set("dealias_en", true);
  // depthCamera->set("pix_cnt_max", setvalue);
  // depthCamera->set("quad_cnt_max", 6);

  depthCamera->set("phase_lin_corr_en", false);

  enum types{TYPE_INT=0, TYPE_UINT, TYPE_BOOL, TYPE_FLOAT};
  std::map<String, uint8_t> params_type;
  params_type.insert(std::pair<String, uint8_t>("unambiguous_range", TYPE_UINT));
  params_type.insert(std::pair<String, uint8_t>("dealiased_ph_mask", TYPE_INT));
  params_type.insert(std::pair<String, uint8_t>("ma", TYPE_UINT));
  params_type.insert(std::pair<String, uint8_t>("mb", TYPE_UINT));
  params_type.insert(std::pair<String, uint8_t>("pix_cnt_max", TYPE_UINT));
  params_type.insert(std::pair<String, uint8_t>("quad_cnt_max", TYPE_INT));
  params_type.insert(std::pair<String, uint8_t>("sub_frame_cnt_max", TYPE_INT));
  params_type.insert(std::pair<String, uint8_t>("hdr_scale", TYPE_UINT));
  params_type.insert(std::pair<String, uint8_t>("pixel_data_size", TYPE_INT));
  params_type.insert(std::pair<String, uint8_t>("op_data_arrange_mode", TYPE_INT));
  params_type.insert(std::pair<String, uint8_t>("dealias_en", TYPE_BOOL));
  params_type.insert(std::pair<String, uint8_t>("phase_lin_corr_period", TYPE_INT));
  params_type.insert(std::pair<String, uint8_t>("phase_lin_corr_en", TYPE_BOOL));
  params_type.insert(std::pair<String, uint8_t>("mod_freq1", TYPE_FLOAT));
  params_type.insert(std::pair<String, uint8_t>("mod_freq2", TYPE_FLOAT));
  params_type.insert(std::pair<String, uint8_t>("phase_corr_1", TYPE_INT));
  params_type.insert(std::pair<String, uint8_t>("phase_corr_2", TYPE_INT));
  params_type.insert(std::pair<String, uint8_t>("intg_duty_cycle", TYPE_UINT));
  params_type.insert(std::pair<String, uint8_t>("coeff_illum", TYPE_INT));
  params_type.insert(std::pair<String, uint8_t>("coeff_sensor", TYPE_INT));
  params_type.insert(std::pair<String, uint8_t>("tillum_calib", TYPE_UINT));
  params_type.insert(std::pair<String, uint8_t>("calib_prec", TYPE_BOOL));

  for(auto itr = params_type.begin(); itr != params_type.end(); ++itr) { 
    bool err{false};
    if(itr->second == TYPE_INT){
      int value{0};
      if(!depthCamera->get(itr->first, value))
        std::cerr<<itr->first<<": Could not read"<<std::endl;
      else
        std::cerr<<itr->first<<": "<<value<<std::endl;
    }

    if(itr->second == TYPE_UINT){
      uint value{0};
      if(!depthCamera->get(itr->first, value))
        std::cerr<<itr->first<<": Could not read"<<std::endl;
      else
        std::cerr<<itr->first<<": "<<value<<std::endl;
    }

    if(itr->second == TYPE_BOOL){
      bool value{0};
      if(!depthCamera->get(itr->first, value))
        std::cerr<<itr->first<<": Could not read"<<std::endl;
      else
        std::cerr<<itr->first<<": "<<value<<std::endl;
    }
    if(itr->second == TYPE_FLOAT){
      float value{0};
      if(!depthCamera->get(itr->first, value))
        std::cerr<<itr->first<<": Could not read"<<std::endl;
      else
        std::cerr<<itr->first<<": "<<value<<std::endl;
    }

  }/*
  // std::vector<uint> values{3072,1170,128,192,256,320,384,448,512,576,640,704,768,832,896,960,3072,1170,128,192,256,320,384,448,512,576,640,704,768,832,896,960};
  for(uint8_t i=0; i<2; ++i){
    for(uint8_t ii=0; ii<16; ++ii){
      // uint value{values[16*i+ii]};
      uint value{0};
      String param{"phase_lin_coeff"};
      param = param +std::to_string(i)+"_"+std::to_string(ii);
      if(depthCamera->set(param, value))
        std::cerr<< param <<": "<<value<<std::endl;
      else
        std::cerr<< param <<": Could not set"<<std::endl;
    }
  }
  for(uint8_t i=0; i<2; ++i){
    for(uint8_t ii=0; ii<16; ++ii){
      uint value{0};
      String param{"phase_lin_coeff"};
      param = param +std::to_string(i)+"_"+std::to_string(ii);
      if(depthCamera->get(param, value))
        std::cerr<< param <<": "<<value<<std::endl;
      else
        std::cerr<< param <<": Could not read"<<std::endl;
    }
  }*/
  std::cerr<<"==================================================\n";
  // return 0;

  // FrameRate fps;
  // fps.numerator = 90;
  // fps.denominator = 1;
  // depthCamera->setFrameRate(fps);


  if(!depthCamera)
  {
    std::cerr << "Could not load depth camera for device " << toConnect->id() << std::endl;
    return -1;
  }

  if(!depthCamera->isInitialized())
  {
    std::cerr << "Depth camera not initialized for device " << toConnect->id() << std::endl;
    return -1;
  }
  
  std::cerr << "Successfully loaded depth camera for device " << toConnect->id() << std::endl;
  
  int count = 0;
  
  TimeStampType lastTimeStamp = 0;

  if (type == "raw")
  {
    depthCamera->registerCallback(DepthCamera::FRAME_RAW_FRAME_UNPROCESSED, [&](DepthCamera &dc, const Frame &frame, DepthCamera::FrameType c) {
      const RawDataFrame *d = dynamic_cast<const RawDataFrame *>(&frame);

      if (!d)
      {
        std::cerr << "Null frame captured? or not of type RawDataFrame" << std::endl;
        return;
      }

      /*std::cerr << "Capture frame " << d->id << "@" << d->timestamp;
      if (lastTimeStamp != 0)
        std::cerr << " (" << 1E6 / (d->timestamp - lastTimeStamp) << " fps)";
      std::cerr << std::endl;*/

      lastTimeStamp = d->timestamp;

      //send data through the pipe
      std::vector<unsigned char> buffer0(d->data.size());
      memcpy(&buffer0[0], &d->data[0], d->data.size());
      is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffer0[0]), buffer0.size());
      std::cout << is.rdbuf();

      // count++;

      if (count >= frameCount)
        dc.stop();
    });
  } 
  else if (type == "raw_processed")
  {
    depthCamera->registerCallback(DepthCamera::FRAME_RAW_FRAME_PROCESSED, [&](DepthCamera &dc, const Frame &frame, DepthCamera::FrameType c) {
      const ToFRawFrame *d = dynamic_cast<const ToFRawFrame *>(&frame);

      if (!d)
      {
        std::cerr << "Null frame captured? or not of type ToFRawFrame" << std::endl;
        return;
      }

      /*std::cerr << "Capture frame " << d->id << "@" << d->timestamp;
      if (lastTimeStamp != 0)
        std::cerr << " (" << 1E6 / (d->timestamp - lastTimeStamp) << " fps)";
      std::cerr << std::endl;*/

      lastTimeStamp = d->timestamp;
      
      //send data through the pipe
      std::vector<unsigned char> buffer0(d->phaseWordWidth()*d->size.width*d->size.height);
      std::vector<unsigned char> buffer1(d->amplitudeWordWidth()*d->size.width*d->size.height);
      std::vector<unsigned char> buffer2(d->ambientWordWidth()*d->size.width*d->size.height);
      std::vector<unsigned char> buffer3(d->flagsWordWidth()*d->size.width*d->size.height);
      memcpy(&buffer0[0], &d->phase()[0], d->phaseWordWidth()*d->size.width*d->size.height);
      memcpy(&buffer1[0], &d->amplitude()[0], d->amplitudeWordWidth()*d->size.width*d->size.height);
      memcpy(&buffer2[0], &d->ambient()[0], d->ambientWordWidth()*d->size.width*d->size.height);
      memcpy(&buffer3[0], &d->flags()[0], d->flagsWordWidth()*d->size.width*d->size.height);
      is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffer0[0]), buffer0.size());
      std::cout << is.rdbuf();
      is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffer1[0]), buffer1.size());
      std::cout << is.rdbuf();
      is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffer2[0]), buffer2.size());
      std::cout << is.rdbuf();
      is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffer3[0]), buffer3.size());
      std::cout << is.rdbuf();

      // count++;

      if (count >= frameCount)
        dc.stop();
    });
  }
  else if (type == "depth")
  {
    depthCamera->registerCallback(DepthCamera::FRAME_DEPTH_FRAME, [&](DepthCamera &dc, const Frame &frame, DepthCamera::FrameType c) {
      const DepthFrame *d = dynamic_cast<const DepthFrame *>(&frame);

      if (!d)
      {
        std::cerr << "Null frame captured? or not of type DepthFrame" << std::endl;
        return;
      }

      /*std::cerr << "Capture frame " << d->id << "@" << d->timestamp;
      if (lastTimeStamp != 0)
        std::cerr << " (" << 1E6 / (d->timestamp - lastTimeStamp) << " fps)";
      std::cerr << std::endl;*/

      lastTimeStamp = d->timestamp;

      // count++;
      //send data through the pipe
      std::vector<unsigned char> buffer0(1*sizeof(float)*d->size.width*d->size.height);
      std::vector<unsigned char> buffer1(1*sizeof(float)*d->size.width*d->size.height);
      memcpy(&buffer0[0], &d->depth[0], sizeof(float)*d->size.width*d->size.height);
      memcpy(&buffer1[0], &d->amplitude[0], sizeof(float)*d->size.width*d->size.height);
      is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffer0[0]), buffer0.size());
      std::cout << is.rdbuf();
      is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffer1[0]), buffer1.size());
      std::cout << is.rdbuf();

      // count++;

      if (count >= frameCount)
        dc.stop();
    });
  }
  else if (type == "pointcloud")
  {
    depthCamera->registerCallback(DepthCamera::FRAME_XYZI_POINT_CLOUD_FRAME, [&](DepthCamera &dc, const Frame &frame, DepthCamera::FrameType c) {
      const XYZIPointCloudFrame *d = dynamic_cast<const XYZIPointCloudFrame *>(&frame);

      if (!d)
      {
        std::cerr << "Null frame captured? or not of type XYZIPointCloudFrame" << std::endl;
        return;
      }

      /*std::cerr << "Capture frame " << d->id << "@" << d->timestamp;
      if (lastTimeStamp != 0)
        std::cerr << " (" << 1E6 / (d->timestamp - lastTimeStamp) << " fps)";
      std::cerr << std::endl;*/

      lastTimeStamp = d->timestamp;

      //send data through the pipe
      std::vector<unsigned char> buffer0(sizeof(IntensityPoint)*d->points.size());
      memcpy(&buffer0[0], &d->points[0], sizeof(IntensityPoint)*d->points.size());
      is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffer0[0]), buffer0.size());
      std::cout << is.rdbuf();

      // count++;

      if (count >= frameCount)
        dc.stop();
    });
  }
  
  if(depthCamera->start())
  {
    FrameRate r;
    if(depthCamera->getFrameRate(r))
      logger(LOG_INFO) << "Capturing at a frame rate of " << r.getFrameRate() << " fps" << std::endl;
    depthCamera->wait();
  }
  else
    std::cerr << "Could not start the depth camera " << depthCamera->id() << std::endl;

  return 0;
}