import threading


def run(f):

    def execute(*a, **kwargs):
        threading.Thread(target=f, args=a, kwargs=kwargs).start()

    return execute