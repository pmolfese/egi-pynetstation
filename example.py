from eci.NetStation import NetStation
from eci.eci import package_event
from time import sleep

from argparse import ArgumentParser

def main():
    p = ArgumentParser(description="Demonstrate NetStation Interface")
    p.add_argument('mode', choices=['local', 'amp'])
    args = p.parse_args()

    if args.mode == 'local':
        IP = '127.0.0.1'
        port = 9885
    elif args.mode == 'amp':
        IP = '10.10.10.42'
        port = 55513
    else:
        raise RuntimeError('Something strange has occured')

    eci_client = NetStation(IP, port)
    eci_client.connect()
    eci_client.begin_rec()
    sleep(3)
    eci_client.send_event(
        3.0,
        event_type='text',
        data={'abcd': 1234},
    )
    eci_client.resync()
    sleep(4)
    eci_client.send_event(
        4.0,
        event_type='hipm',
        data={'hola': 'Greetings, Pete!'},
    )
    eci_client.end_rec()
    eci_client.disconnect()
if __name__ == '__main__':
    main()