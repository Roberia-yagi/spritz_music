import whisper

def compare_models(file_path):
    models = ['tiny', 'base', 'small', 'medium', 'large']
    for model_name in models:
        model = whisper.load_model(model_name)
        result = model.transcribe(file_path, fp16=False)
        print(f'model: {model_name} \n {result["text"]}')
        print('--------')

def main():
    file_path = 'materials/music.wav'
    compare_models(file_path)

if __name__=='__main__':
    main()