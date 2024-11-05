# Conv2D-LSTM
Combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks 

 Warning: run only Hm Node

developed in indian institute of technology kharagpur

by Subhadeep Maishal  


examples : 
![Figure](https://github.com/subhadeep-maishal/Conv2D-LSTM/blob/main/conv2D%2BLSTM.png) 



# 🌟 Conv2D + LSTM Architecture 🌟

This section describes the combined architecture of **Convolutional Neural Networks (CNNs)** and **Long Short-Term Memory (LSTM)** networks, ideal for tasks that involve both spatial and temporal data analysis.

## 🖼️ Convolutional Layer (Conv2D)

The **Conv2D** layer is responsible for extracting spatial features from the input data. Its output can be expressed mathematically as:

```plaintext
Output_conv = Conv2D(Input, Filters, Kernel Size, Strides, Padding)
```

For a single channel input, the output can be described as:

```plaintext
Output_conv[i, j, k] = ∑(m=0 to H-1) ∑(n=0 to W-1) Input[i+m, j+n, c] * Filters[m, n, c, k] + Bias[k]
```

### Where:
- **`H`** and **`W`** are the height and width of the kernel.
- **`c`** iterates over the number of input channels.
- **`k`** is the index of the output channel.

---

## 🧠 LSTM Layer

The **LSTM** layer processes sequences over time, making it perfect for temporal analysis. The equations for a single LSTM cell are as follows:

### 🔑 Forget Gate

```plaintext
f_t = σ(W_f ⋅ [h_{t-1}, x_t] + b_f)
```

### ➕ Input Gate

```plaintext
i_t = σ(W_i ⋅ [h_{t-1}, x_t] + b_i)
```

### 📝 Candidate Memory Cell

```plaintext
C~_t = tanh(W_C ⋅ [h_{t-1}, x_t] + b_C)
```

### 🔄 Cell State Update

```plaintext
C_t = f_t ⋅ C_{t-1} + i_t ⋅ C~_t
```

### 📤 Output Gate

```plaintext
o_t = σ(W_o ⋅ [h_{t-1}, x_t] + b_o)
```

### 📥 Hidden State Update

```plaintext
h_t = o_t ⋅ tanh(C_t)
```

### Where:
- **`W`** and **`b`** represent the weights and biases.
- **`x_t`** is the input at time **`t`**.
- **`h_t`** is the hidden state output.

---

## 🔗 Combined Equation

The overall processing of the architecture can be expressed as:

```plaintext
h_t = LSTM(Flatten(Conv2D(Input)))
```

This architecture effectively leverages spatial feature extraction from the **Conv2D** layers followed by temporal sequence modeling through the **LSTM** layers, providing powerful capabilities for analyzing complex data.

---

## 🚀 Conclusion

By combining **Conv2D** and **LSTM**, * harness the strengths of both convolutional and recurrent networks, making this architecture suitable for a variety of applications, including:

- Oceanic predictions
- Time series forecasting 📈
- Sequential data prediction 🔮

Feel free to explore the code and examples provided in this repository to see this architecture in action!
