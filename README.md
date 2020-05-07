# tep-chebyshev
Provide source code for the paper *"Dimension Reduction of thermoelectric properties using barycentric polynomial interpolation at Chebyshev nodes"* by Jaywan Chung, Byungki Ryu and SuDong Park.

## Python package requirements
The source code is tested with the following 3rd party libraries along with the Python programming language:
- Python==3.6.8
- numpy==1.16.3
- pandas==0.24.2
- scipy==1.2.1
- sqlite==3.28.0

## How to reproduce the results
- **Figures** in the paper can be drawn by executing the python code starting with "plot_".\
  But before executing the code, one may need to execute other python code to obtain required data to draw.\
  Refer to *File description* section below for accurate information.
- **Table** values of *Supplementary Information* can be obtained by executing the python code starting with "table_".\
  But before executing the code, one may need to execute other python code to obtain required data.\
  Refer to *File description* section below for accurate information.

## File description
### Auxiliary files
- tep_20180409.db\
  A sqlite database file containing **TEP data**.
- data_info.csv\
  Explain doi, material type and base material of the TEPs used in the experiment.\
  This file is used to check which ids are used for experiments; the TEPs corresponding to the ids in the file are loaded from the db file.\
  A more detailed information is given in *Supplementary Information* of the paper.

### Python code
- libs/pykeri/\
	A package to manage TEP database.
- utils.py\
	Barycentric polynomial interpolator, and one-dimensional thermoelectric equation solver.
- test_barycentric_cheb.py\
	Test the barycentric polynomial interpolation at Chebyshev nodes in utils.py.
- test_barycentric_equi.py\
	Test the barycentric polynomial interpolation at equidistant nodes in utils.py.
- accuracy_barycentric_cheb_n_nodes.py\
	Create a csv file showing the interpolation accuracy of Chebyshev nodes.\
	Type "python accuracy_barycentric_cheb_n_nodes.py [NUM_NODES]" in the command line to execute it.\
	Default NUM_NODES is 11.
- accuracy_barycentric_equi_n_nodes.py\
	Create a csv file showing the interpolation accuracy of equidistant nodes.\
	Type "python accuracy_barycentric_equi_n_nodes.py [NUM_NODES]" in the command line to execute it.\
	Default NUM_NODES is 11.
- plot_barycentric_equi_n_nodes_for_paper.py\
	Plot **the top of Figure 1** of the paper.
- plot_barycentric_cheb_n_nodes_for_paper.py\
	Plot **the bottom of Figure 1** of the paper.
- plot_accuracy_barycentric_no_error_bar.py\
	Plot **the top of Figure 2** and **the top of Figure 3** of the paper.\
	This script assumes the interpolation accuracy for each node is already computed by using "accuracy_barycentric_cheb_n_nodes.py" and "accuracy_barycentric_equi_n_nodes.py".\
	In the repository, the assumption is already fulfilled with the data in "results/accuracy" directory.
- plot_accuracy_barycentric_error_bar.py\
	Plot **the bottom of Figure 2** and **the bottom of Figure 3** of the paper.\
	This script assumes the interpolation accuracy for each node is already computed by using "accuracy_barycentric_cheb_n_nodes.py".\
	In the repository, the assumption is already fulfilled with the data in "results/accuracy" directory.
- performance_exact.py\
	Compute the performance of thermoelectric modules using the exact TEPs, and save it to a csv file.\
	The exact TEPs are assumed to be spline curves: 2nd-order spline for Seebeck coefficient, piecewise linear curves for electrical resistivity and thermal conductivity.
- performance_barycentric_cheb_n_nodes.py\
	Compute the performance of thermoelectric modules using the barycentric polynomial interpolation of TEPs at Chebyshev nodes.\
	Then create a csv file showing the performance.\
	Type "python performance_barycentric_cheb_n_nodes.py \[NUM_NODES\]" in the command line to execute it.\
	Default NUM_NODES is 11.
- plot_performance_barycentric_error_bar.py\
	Plot **Figure 4** of the paper.\
	This script assumes the performance accuracy for each node is already computed by using "performance_exact.py" and "performance_barycentric_cheb_n_nodes.py".\
	In the repository, the assumption is already fulfilled with the data in "results/performance" directory.
- accuracy_barycentric_cheb_n_nodes_with_noise.py\
	Create a csv file showing the interpolation accuracy of Chebyshev nodes under noise.\
	Type "python accuracy_barycentric_cheb_n_nodes_with_noise.py \[NUM_NODES\] \[NOISE_PERCENT\]" in the command line to execute it.\
	Default NUM_NODES is 11. Default NOISE_PERCENT is 0.
- plot_accuracy_vs_noise.py\
	Plot **Figure 5** of the paper.\
	This script assumes the interpolation accuracy for each node under noise is already computed by using "accuracy_barycentric_cheb_n_nodes_with_noise.py".\
	In the repository, the assumption is already fulfilled with the data in "results/accuracy_noise" directory.
- performance_barycentric_cheb_n_nodes_with_noise.py\
	Compute the performance of thermoelectric modules using the barycentric polynomial interpolation of TEPs at Chebyshev nodes under noise. Then create a csv file showing the performance.\
	Type "python performance_barycentric_cheb_n_nodes_with_noise.py \[NUM_NODES\] \[NOISE_PERCENT\]" in the command line to execute it.
	Default NUM_NODES is 11. Default NOISE_PERCENT is 0.
- plot_performance_vs_noise.py\
	Plot **Figure 6** of the paper.\
	This script assumes the performance accuracy for each node under noise is already computed by using "performance_exact.py" and "performance_barycentric_cheb_n_nodes_with_noise.py".\
	In the repository, the assumption is already fulfilled with the data in "results/performance/performance_exact.csv" file and "results/performance_noise" directory.
- plot_barycentric_cheb_n_nodes_for_paper_troubled_case.py\
	Plot **Figure 7** of the paper.
- plot_barycentric_cheb_n_nodes_for_paper_troubled_case_remedy.py\
	Plot **Figure 8** of the paper.
- table_accuracy.py\
	Make **Tables 2, 3, 4, 5** in *Supplementary Information* of the paper.\
	This script assumes the interpolation accuracy for each node is already computed by using "accuracy_barycentric_cheb_n_nodes.py" and "accuracy_barycentric_equi_n_nodes.py".\
	In the repository, the assumption is already fulfilled with the data in "results/accuracy" directory.
- table_performance.py\
	Make **Tables 6, 7** in *Supplementary Information* of the paper.\
	This script assumes the performance accuracy for each node is already computed by using "performance_exact.py" and "performance_barycentric_cheb_n_nodes.py".\
	In the repository, the assumption is already fulfilled with the data in "results/performance" directory.
- table_accuracy_vs_noise.py\
	Make **Tables 8, 9, 10** in *Supplementary Information* of the paper.\
	This script assumes the interpolation accuracy for each node under noise is already computed by using "accuracy_barycentric_cheb_n_nodes_with_noise.py".\
	In the repository, the assumption is already fulfilled with the data in "results/accuracy_noise" directory.
- table_performance_vs_noise.py\
	Make **Tables 11, 12, 13, 14, 15, 16** in *Supplementary Information* of the paper.\
	This script assumes the performance accuracy for each node under noise is already computed by using "performance_exact.py" and "performance_barycentric_cheb_n_nodes_with_noise.py".\
	In the repository, the assumption is already fulfilled with the data in "results/performance_noise" directory.
