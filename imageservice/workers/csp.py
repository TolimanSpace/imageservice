from ..ErrorCode import ErrorCode, result
from ...csp import csp

# CSP Configuration
csp.init(
    10, "service_name", "hostname", "model", "revision"
)  # Adjust the parameters accordingly
csp.rdp_init()
csp.service_start()


# CSP Listener Function
def csp_listener(shared_status):
    csp.initialise()
    address = 0
    listener = csp.Listener(address)
    # listener.register_callback(lambda msg: print(f"Received message: {msg}"))
    listener.register_callback(proccess_cmds)
    listener.start()

    # Listener loop - modify as needed
    while True:
        pass


# Read or update shared_status based on CSP messages
# ...
def proccess_cmds(msg: (bytes)):

    sender = csp.Sender(address)
    ts=telemetery_switch()
    result = ts.result_for_cmd(msg)
    lasterror = replymethod(result)


# CSP Sender Function
# def csp_sender(shared_status):
#     # csp.initialise()
#     address = 0
#     sender = csp.Sender(address)
#     sender.send(b"Hello World")



class telemetery_switch(object):
    cmds = {
        "0x01": ("BlockSize", 1),
        "0x10": ("ResolutionX", 2),
        "0x11": ("ResolutionY", 2),
        "0x12": ("ExposureMode", 2),
        "0x13": ("ExposureTime", 4),
        "0x14": ("ShutterSpeedValue", 4),
        "0x15": ("MeteringMode", 2),
        "0x16": ("UnixTime", 4),
        "0x17": ("ISOSpeedRatings", 2),
        "0x18": ("UnixTimeSubSec", 4),
        "0x21": ("NumBlocksJPEG", 4),
        "0x22": ("ImageSizeJPEG", 4),
        "0x23": ("ImageCRCJPEG", 4),
        "0x31": ("NumBlocksRAW", 4),
        "0x32": ("ImageSizeRAW", 4),
        "0x33": ("ImageCRCRAW", 4),
    }

    def __init__(self, so):
        self.so = so

    def result_for_cmd(self, argument: bytes, camdata_attr: str):
        """
        Dispatch method for telemetry commands.
            :param self: self
            :param argument: byte array of raw cmds recived  over csp
        """

        cmd_revc = "0x{:02x}".format(argument)
        (tag, numbytes) = self.cmds[cmd_revc]
        try:
            val = int(
                getattr(self.so, camdata_attr)[
                    getattr(self.so, camdata_attr + "_index")
                ][tag]
            )
            return result(val, "u" + str(numbytes * 8))
        except IndexError as err:
            val = 0
            result(ErrorCode.ImageIndexNotValid)


def printcmds(argument, preamble=""):
    """
    Helper method to print bytes array received from i2c as hex values.
        :param argument: raw cmds from i2c in byte array
        :param preamble='': text to print before printing argumemt.
    """
    print(
        str(datetime.now())
        + " | "
        + preamble
        + "cmd=["
        + " ".join("0x{:02x}".format(x) for x in argument)
        + "]"
    )


def replymethod(self,sender, cmdresult: result):

    # result=namedtuple('Result', ['value','returntype'])
    if cmdresult:
        if type(cmdresult.value) is int:

            sender.send(cmdresult.value)
        elif type(cmdresult.value) is ErrorCode:
            self.lasterror = cmdresult.value
            return
        else:
            print("not impl")
            self.lasterror = ErrorCode.UnknownCommand
            return

        self.lasterror = ErrorCode.Ok
