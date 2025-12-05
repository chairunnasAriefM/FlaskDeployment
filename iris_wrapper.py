class IrisWrapper:
    def __init__(self, model, scaler, encoder):
        self.model = model
        self.scaler = scaler
        self.encoder = encoder

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_pred_num = self.model.predict(X_scaled)
        y_pred_label = self.encoder.inverse_transform(y_pred_num)
        return y_pred_label
