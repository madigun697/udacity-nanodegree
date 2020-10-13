#### Long Short-Term Memory(LSTM) Network

##### Structure of LSTM cell

![image](https://user-images.githubusercontent.com/8471958/95925175-f9e40a00-0d6d-11eb-9129-bc057363ec99.png)

![image](https://user-images.githubusercontent.com/8471958/95925205-0ff1ca80-0d6e-11eb-8514-a5a0a1319fb2.png)

- LSTM cell has four different gates
  1. Learn Gate
     ![image](https://user-images.githubusercontent.com/8471958/95925264-31eb4d00-0d6e-11eb-809e-6d6c69183e4c.png)
     - Learn gate determines what memory has to learn or has to ignore in the short-term memory
     - The previous short-term memory is combined to current event
     - The ignore factor($$i_t$$) is calculated by previous short-term memory($$STM_{t-1}$$) and current event($$E_t$$)
  2. Forget Gate
     ![image](https://user-images.githubusercontent.com/8471958/95925288-416a9600-0d6e-11eb-9641-7209b9246034.png)
     - Forget gate determines what memory has to remain or has to forget in the long-term memory
     - The forget factor($$f_t$$) is calculated by previous short-term memory($$STM_{t-1}$$) and current event($$E_t$$)
  3. Remember Gate
     ![image](https://user-images.githubusercontent.com/8471958/95925316-56dfc000-0d6e-11eb-9879-dad4156e6d19.png)
     - Remember gate determines the next long-term memory to add up results of forget gate and learn gate togeter
     - The result of forget gate = $$LTM_{t-1} \cdot f_t = LTM_{t-1} \cdot \sigma(W_f[STM_{t-1}, E_t] + b_f)$$
     - The result of learn gate = $$N_i \cdot i_t = tanh(W_n[STM_{t-1}, E_t] + b_n) \cdot \sigma(W_i[STM_{t-1}, E_t]+b_i)$$
     - The result of remember gate($$LTM_t$$) = $$LTM_{t-1} \cdot f_t + N_i \cdot i_t$$
  4. Use Gate
     ![image](https://user-images.githubusercontent.com/8471958/95925326-65c67280-0d6e-11eb-86d1-5fa74c66bd2d.png)
     - Use gate determines the next short-term memory