from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Generate some sample data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.randn(100)

# Plot the data to visualize it


# plt.scatter(X, y)

# Fit the linear regression model
X = X.reshape((-1, 1))
reg = LinearRegression().fit(X, y)

# Print the slope and intercept
print("Slope: ", reg.coef_[0])
print("Intercept: ", reg.intercept_)

# Plot the regression line
fig, ax = plt.subplots()
ax.scatter(X,y)
ax.plot(X, reg.predict(X), color='r')
ax.set_xlabel('X')
ax.set_ylabel('y')
st.subheader('Scatter Plot')
st.pyplot(fig)


plt.show()

st.write("Slope: ", reg.coef_[0])
st.write("Intercept: ", reg.intercept_)

    
