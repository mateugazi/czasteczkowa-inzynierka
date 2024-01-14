<script>
	export let trainingResults;
	let header = trainingResults[0];
	let rows = trainingResults.slice(1);
</script>

<div class="overflow-x-auto">
	<table class="table table-zebra table-xs">
		<!-- head -->
		<thead>
			<tr>
				<th></th>
				{#each header as columnName}
					<th>{columnName}</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each rows as row, i}
				<tr>
					<th>{i}</th>
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
</style>
