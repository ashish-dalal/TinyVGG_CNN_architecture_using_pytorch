import argparse

def main():
    parser = argparse.ArgumentParser(description='temp file to test argparse')

    parser.add_argument('-i','--integer', type=int, default=15, help='int value')

    args, unknown_args = parser.parse_known_args()

    integer_bool = (args.integer != 15)  or ('--integer' in unknown_args)

    integer_value = args.integer

    print(f"integer provided: {integer_bool}, integer value: {integer_value}")

if __name__ ==  "__main__":
    main()