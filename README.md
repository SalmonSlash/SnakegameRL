# SnakegameRL

​This project aims to explore Reinforcement Learning (RL) by developing an AI that plays the classic Snake game. The motivation stems from personal challenges in playing Google's Snake game and a desire to observe the AI's learning progress. Currently, the system can train models using simulated environments and deploy them to play the actual game. However, performance is limited, possibly due to slow response times or insufficient model training.​

Core Concept:

The approach involves training the AI in a simulated environment with 11-dimensional input features. A standard linear model is used for training. The trained model is then applied to control the Snake game in the browser by capturing screen images, processing them to identify the game grid, and using the model to make real-time decisions via keyboard inputs (WASD keys).​

Next Steps:

User Guide: Develop comprehensive documentation detailing setup, usage instructions, and configuration options to assist users in effectively utilizing the system.​
Development Philosophy: Articulate the design principles and decision-making processes that guided the project's development, providing clarity on the chosen methodologies and approaches.
Code Comments: Enhance code readability and maintainability by adding descriptive comments, ensuring that the purpose and functionality of code segments are clear to current and future developers.​
Unit Testing: Implement unit tests to validate individual components of the system, ensuring reliability and facilitating easier debugging and future development.