<script>
	import { onMount } from "svelte";
	import SamplePlot from "../../plots/SamplePlot.svelte";
	import { store } from "../../store/store";
	import PredictionsTable from "../PredictionsTable.svelte";
	import SummaryDetails from "../SummaryDetails.svelte";
	import { scrollToTop } from "../../helpers/scrollToTop";

  let predictions;

  onMount(() => {
    scrollToTop();
  });

  const unsubscribe = store.subscribe((state) => {
    predictions = state.predictions;
  });

</script>

<div>
  <PredictionsTable predictions={predictions} /> 
  <div class="hero-content flex-col lg:flex-row">
    <SamplePlot />
    <SummaryDetails />
  </div>
</div>