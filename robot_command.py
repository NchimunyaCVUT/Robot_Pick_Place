import asyncua.sync as opcua
import asyncua.ua as uatype
import logging




class  RobotCommand:
    """Class to handle robot commands via OPC UA"""
    
    def __init__(self, logging_config):
        """Initialize the RobotCommand class"""

        
        PLC_OPCUA_URL = 'opc.tcp://10.100.0.120:4840/'
        opcua_client = opcua.Client(PLC_OPCUA_URL)
        opcua_client.connect()
    
        types = opcua_client.load_data_type_definitions()
        workplace_ns_uri = "http://w4.ti40.cz"
        robots_ns_uri = "http://robots.ti40.cz"
        transport_ns_uri = "http://transportation.ti40.cz"
        workplace_ns_idx = opcua_client.get_namespace_index(workplace_ns_uri)
        robots_ns_idx = opcua_client.get_namespace_index(robots_ns_uri)
        transport_ns_idx = opcua_client.get_namespace_index(transport_ns_uri)

        ########################################
        # Read nodes from server using node IDs:
        # Robot nodes:
        node_robot = opcua_client.get_node(f'ns={robots_ns_idx};s=Cell.Robots.R32')
        node_robot_coom_err = opcua_client.get_node(f'ns={robots_ns_idx};s=Cell.Robots.R32.CommunicationError') # Bool
        node_robot_powered_on = opcua_client.get_node(f'ns={robots_ns_idx};s=Cell.Robots.R32.PoweredON') # Bool
        node_robot_busy = opcua_client.get_node(f'ns={robots_ns_idx};s=Cell.Robots.R32.Busy') # Bool
        node_robot_is_homed = opcua_client.get_node(f'ns={robots_ns_idx};s=Cell.Robots.R32.IsHomed') # Bool
        node_robot_position = opcua_client.get_node(f'ns={robots_ns_idx};s=Cell.Robots.R32.Position') # POS6D
        node_robot_position_valid = opcua_client.get_node(f'ns={robots_ns_idx};s=Cell.Robots.R32.PositionValid') # Bool
        node_robot_gripper_active = opcua_client.get_node(f'ns={robots_ns_idx};s=Cell.Robots.R32.GripperActive') # Bool