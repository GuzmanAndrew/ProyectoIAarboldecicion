# main.py
from src.data_loader import load_data, check_data
from src.model import train_model, evaluate_model

def main():
    # mostra los datos
    # check_data()

    # Cargar y dividir los datos
    X_train, X_test, y_train, y_test = load_data()

    # Entrenar el modelo
    model = train_model(X_train, y_train)

    # Evaluar el modelo
    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy

if __name__ == '__main__':
    main()
