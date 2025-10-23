import main.py
import sys
import os

from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient

from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, DeviceManager_pb2, VisionConfig_pb2

# Initalize K3N 
def initalizeKinova():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()


# Getting the ID of the K3N color sensor (Camera). This was taken from the API Github
def example_vision_get_device_id(device_manager):
    vision_device_id = 0
    
    # Getting all device routing information (from DeviceManagerClient service)
    all_devices_info = device_manager.ReadAllDevices()

    vision_handles = [ hd for hd in all_devices_info.device_handle if hd.device_type == DeviceConfig_pb2.VISION ]
    if len(vision_handles) == 0:
        print("Error: there is no vision device registered in the devices info")
    elif len(vision_handles) > 1:
        print("Error: there are more than one vision device registered in the devices info")
    else:
        handle = vision_handles[0]
        vision_device_id = handle.device_identifier
        print("Vision module found, device Id: {0}".format(vision_device_id))

    return vision_device_id

def main():
    
    
    # Create connection to the device and get the router
    

        if vision_device_id != 0:
            example_routed_vision_get_intrinsics(vision_config, vision_device_id)
            example_routed_vision_set_intrinsics(vision_config, vision_device_id)

if __name__ == "__main__":
    main()