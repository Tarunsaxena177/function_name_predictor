import joblib
import sys
import os

def predict_function(input_text):
    if not os.path.exists('model.pkl'):
        return "Error: Model not trained. Run train.py first."

    # Load assets
    model = joblib.load('model.pkl')
    encoder = joblib.load('encoder.pkl')

    # Prediction
    prediction_encoded = model.predict([input_text])
    function_name = encoder.inverse_transform(prediction_encoded)
    
    return function_name[0]

if __name__ == "__main__":
    # Default example from Assignment Section 4
    test_input = "Adds two integers int a int b return int keywords add sum"
    
    if len(sys.argv) > 1:
        test_input = " ".join(sys.argv[1:])

    result = predict_function(test_input)
    
    print(f"\n--- Inference Demo ---")
    print(f"Input Metadata: {test_input}")
    print(f"Predicted Function: {result}")