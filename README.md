GPU_4Polygon
============

4p


Similar to the CUDA_brute_triangle project, but this implementation brute-force examines all N choose 4 2-D points and finds which polygon(s) contain the greatest number of points inside.

Not quite as fast as the 3 point version, but still kills an i-7 3770k 3.9Ghz CPU (full -O2 optimizations).


Optimal Polygon Running Time comparison:
---
<table>
<tr>
    <th>NumPoints</th><th>NumCombosEvaluated</th><th> 3.9 Ghz CPU time </th><th> 1.01 Ghz GPU time </th><th> CUDA Speedup</th>
</tr>
    <tr>
    <td> 100</td><td>3,921,225</td><td> 5,672 ms </td><td> 16ms </td><td> 354.5x</td>
  </tr
  <tr>
    <td> 200</td><td>64,684,950</td><td> 196,500ms </td><td> 422 ms </td><td> 465.6x </td>
</tr>
<tr>
    <td> 300</td><td>330,791,175</td><td> 1,589,806 ms</td><td> 3,130 ms </td><td> 507.9x </td>
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
