from src.controllers.score_manager import ScoreManager


def main():
    """
    Trigger the Scoremanager which again sequentially invokes other modules for the player recognition.
    """
    ScoreManager().run()


if __name__ == '__main__':
    main()
