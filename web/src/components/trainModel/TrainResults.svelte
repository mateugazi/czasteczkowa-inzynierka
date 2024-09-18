<script>
	import { store } from "../../store/store";

	let trainingResults;
	let modelInfo;

	const unsubscribe = store.subscribe((state) => {
		if (state.trainingResults.withModelInfo) {
			modelInfo = state.trainingResults.data.slice(0, 2);
			trainingResults = state.trainingResults.data.slice(2);
		} else {
			trainingResults = state.trainingResults.data;
		}
	});
</script>

<h1 class="text-5xl font-bold">Your results</h1>

{#if modelInfo}<div class="model-info">
		{#each modelInfo as row}
			<div class="stats shadow">
				<div class="stat">
					<div class="stat-title">{row[0]}</div>
					<div class="stat-value">{row[1]}</div>
				</div>
			</div>
		{/each}
	</div>
{/if}
<div class="overflow-x-auto">
	<table class="table table-zebra table-lg">
		<thead>
			<tr>
				<th>Metric</th>
				<th>Score</th>
			</tr>
		</thead>
		<tbody>
			{#each trainingResults as row, i}
				<tr>
					{#each row as element}
						<td>{element}</td>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.overflow-x-auto {
		min-height: 70vh;
	}

	h1 {
		margin: 30px 0;
	}

	.model-info {
		display: flex;
		flex-direction: column;
	}
</style>
