<script>
	import { mapPredictionLabel } from "../helpers/mapPredictionLabel";

	export let predictions = [];
</script>

<div class="overflow-x-auto">
	<table class="table">
		<!-- head -->
		<thead>
			<tr>
				<th></th>
				<th class="bg">Molecule</th>
				<th>Type</th>
			</tr>
		</thead>
		<tbody>
			{#each predictions as prediction, i (i)}
				<tr class="hover">
					<th>{i}</th>
					<td>{prediction.mol}</td>
					<td>{mapPredictionLabel(prediction.predictedClass)}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.overflow-x-auto {
		height: 450px;
		overflow: auto;
	}
</style>
