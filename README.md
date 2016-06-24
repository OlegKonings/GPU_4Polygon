GPU_4Polygon
============

4p


Similar to the CUDA_brute_triangle project, but this implementation brute-force examines all N choose 4 2-D points and finds which polygon(s) contain the greatest number of points inside.

Not quite as fast as the 3 point version, but much faster than a serial implementation. If you want an estimate of a multi-core implementation you can divide the CPU time by the number of cores (not really that simple in real life). Even then much much faster than a multi-core implementation, and this performance despite the fact there is a great deal of 64-bit integer operations which are much slower than 32-bit operations on consumer GPUs.

CPU used= Intel i7 4820k 4.5 GHZ

GPU used= NVIDIA GTX 1080 1.8 Ghz

CUDA 8.0 RC

Windows 7 x64



Optimal Polygon Running Time comparison:
---
<table>
<tr>
    <th>NumPoints</th><th>NumCombosEvaluated</th><th> 4.5 Ghz CPU time </th><th> 1.01 Ghz GPU time </th><th> CUDA Speedup</th>
</tr>
    <tr>
    <td> 100</td><td>3,921,225</td><td> 4,4245 ms </td><td> 9ms </td><td> 471.6x</td>
  </tr
  <tr>
    <td> 200</td><td>64,684,950</td><td> 148,411ms </td><td> 250 ms </td><td> 593.6x </td>
</tr>
<tr>
    
</tr>

</table>

___



Still think this can be improved, but in general this technique works and will run even faster on a GTX780ti, so will post those results soon.

Can handle up to 2000 points, but any more will overflow __constant__ memory. 

Linux users need to change the long long(a) casts to (long long)a . Does need a GPU will compute capability >3.0.

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-60172288-1', 'auto');
  ga('send', 'pageview');

</script>
