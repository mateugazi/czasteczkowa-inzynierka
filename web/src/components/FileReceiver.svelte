<script>
	import { createEventDispatcher } from "svelte";
  let files;

	const dispatch = createEventDispatcher();
	let foo = 'baz'
	let bar = 'qux'
  let result = null;

  async function doPost () {
		// const res = await fetch('http://localhost:3000/get-predictions', {
		// 	method: 'POST',
		// 	body: JSON.stringify({'pozdro': '600'}),
    //   headers: {
    //     "Content-Type": "application/json"
    //   },
		// })
		
		// const json = await res.json();
	  // result = JSON.stringify(json);
    // console.log(result);

    const data = new FormData();
    data.append('file', files[0]);
    const res = await fetch('http://localhost:3000/upload-csv', {
			method: 'POST',
			body: data,
		})
		
		const json = await res.json();
	  result = JSON.stringify(json);
    console.log(result);

    // const data = new FormData();
    // data.append('file', files[0]);
    // const xhr = new XMLHttpRequest();

    // xhr.addEventListener("readystatechange", function () {
    //   if (this.readyState === this.DONE) {
    //     console.log(this.responseText);
    //   }
    // });

    // xhr.open("POST", "http://localhost:3000/upload-csv");

    // xhr.send(data);
  }
</script>

<input accept=".csv" bind:files type="file" class="file-input file-input-bordered file-input-primary w-full max-w-xs" />

{#if files}
  <button class="btn btn-primary btn-lg" on:click={() => doPost()}>Run predictions</button>
{/if}

<style>
  .btn {
    margin-top: 30px;
  }
</style>

