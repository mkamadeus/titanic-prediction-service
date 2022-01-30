from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np

app = FastAPI()
sess = rt.InferenceSession("titanic.onnx")

class Input(BaseModel):
	Pclass : float
	SibSp : float
	Parch : float
	Age : float
	Sex_female : float
	Sex_male : float
	Embarked_C : float
	Embarked_Q : float
	Embarked_S : float

print(sess.get_inputs()[0])
print(sess.get_outputs()[0], sess.get_outputs()[1])

val = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: np.array([[3, 0, 0, 0.263238, 0, 1, 0, 1, 0]], dtype=np.float32)})[0]
print(val)
@app.post('/')
async def root(body : Input):
	input_name = sess.get_inputs()[0].name
	label_name = sess.get_outputs()[0].name
	# pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
