from demo.s1_config import app


def main():
    app.kmeans_evaluate(use_cache=False)


if __name__ == '__main__':
    main()
