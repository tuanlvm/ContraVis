# ContraVis

INTRODUCTION

This is an implementation of ContraVis - contrastive and visual topic modeling for comparing document collections.

Usage:

	perl contravis.pl	--num_topics $num_topics
        			--num_labels $num_labels
				--dim $dim
				--lambda $lambda
        			--varphi $varphi
				--gamma $gamma
        			--sigma_0 $sigma_0
				--EM_iter $EM_iter
				--Quasi_iter $Quasi_iter
				--data $data
				--output_file $output_file

Arguments:

	$num_topics: number of topics
	$num_labels: number of labels (include the common label)
	$dim: number of dimensions
	$lambda: Dirichlet parameter
	$varphi: covariance for Gaussian prior of topic coordinates
	$gamma: covariance for Gaussian prior of document coordinates
	$sigma_0: covariance for Gaussian prior of label coordinates
	$EM_iter: number of iterations for EM
	$Quasi_iter: maximum iterations of Quasi-Newton
	$data: input data
	$output_file: output file
	

+ Data: each line is a document containing ids of words. The first $num_labels numbers constitute a 0-1 vector l telling which labels are assigned to that document. l[i] = 1 if document d has label i.

