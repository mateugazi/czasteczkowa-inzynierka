<script>
	export let parameterLabel;
</script>

<label class="form-control w-full max-w-xs">
	<div class="label">
		<span class="label-text">{parameterLabel}</span>
	</div>
	<input
		type="text"
		placeholder="Type here"
		class="input input-bordered w-full max-w-xs"
		on:change
	/>
</label>
