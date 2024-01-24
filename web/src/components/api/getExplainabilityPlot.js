export const getExplainabilityPlot = async () => {
  await fetch("http://localhost:3000/explainability-plot")
    	.then(function(data){
        return data.blob();
      })
      .then(function(img){
      	var dd = URL.createObjectURL(img);
        document.getElementById('explainability-plot').setAttribute('src', dd);
      })
}