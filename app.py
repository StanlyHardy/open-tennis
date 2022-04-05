from src.controllers.open_tennis import OpenTennis


def main():
    """
    Trigger the OpenTennis which again sequentially invokes other modules for the player recognition.
    """
    OpenTennis().run()


if __name__ == '__main__':
    main()
