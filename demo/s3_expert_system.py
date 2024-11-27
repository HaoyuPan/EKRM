import subprocess

from demo.s1_config import app


def main():
    app.generate_predict_results()
    print(f'Please open the directory to categorize the colors: {app.kmeans_get_classification_dir()}')
    subprocess.Popen(['explorer.exe', str(app.kmeans_get_classification_dir())])


if __name__ == '__main__':
    main()
