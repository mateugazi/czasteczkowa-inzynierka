# CHEML
### AN APPLICATION SUPPORTING DISCOVERING BIOLOGICALLY ACTIVE MOLECULES FOR DRUG DESIGN USING MACHINE LEARNING METHODS
CHEML is a project developed as Batchelor thesis on Politechnika Poznańska by Jakub Cichy, Mateusz Duda, Joanna Działo and Wojciech Majewski
#### Abstract
”CheML” emerges as an innovative web-based application tailored to streamline drug discovery by harnessing machine learning (ML). It integrates a variety of ML algorithms, including decision trees, ensemble methods, and neural networks, to analyze and predict pharmaceutical properties of molecules using datasets like BACE and RORγ. Through a thorough search on hyperparameters the best models have been found and proposed as a baseline for future research. Central to its functionality is the focus on model explainability, balancing the need for accurate predictions with the necessity of transparent AI methodologies. Developed using contemporary technologies like Svelte, Vite for the client interface, Flask for server-side operations, and MongoDB for data management, ”CheML” offers a user-centric and intuitive platform. Its ability to navigate the complex trade-off between accuracy and explainability in ML models positions it as a pivotal tool in pharmaceutical research, paving the way for enhanced understanding of drug efficacy and molecular dynamics, thus revolutionizing the role of AI in drug discovery.

### Application
The CheML application offers a user-friendly interface designed for seamless interaction with machine learning models in drug discovery. The [welcome screen](img/application-overview.png) guides users to select from three core workflows: retraining existing models, training new models from scratch, or obtaining predictions. Each workflow is accessed through a simple card-based interface, ensuring smooth navigation. The retrain option allows users to refine existing models by uploading new data, with results presented in detailed metrics for both classification and regression models. In the [new model training workflow](img/application-train-new-model-response.png), users can customize the model architecture or use default settings, with the application providing support for both novice and advanced users. The [prediction](img/application-predictions-modal.png) workflow delivers results with predicted labels and a SHAP plot, offering insights into feature importance for transparency. The application is designed to balance functionality and simplicity, making machine learning accessible while maintaining robust customization options.
