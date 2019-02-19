use warnings;
use strict;
use Data::Dumper;
use Math::Trig;
use Algorithm::LBFGS;
use Getopt::Long;

our $num_topics = 20;
our $num_labels = 3;
our $data;
our $dim = 2;
our $num_iter = 200;
our $num_quasi_iter = 20;
our $output_file;
our $lambda;
our $varphi;
our $gamma;
our $sigma_0;
our @doc_num_labels;

our @likelihoods;

GetOptions ("num_topics=i" => \$num_topics,
			"num_labels=i" => \$num_labels,
			"dim=i" => \$dim,
            "lambda=f"    => \$lambda,
            "varphi=f"     => \$varphi,
            "gamma=f"   => \$gamma,
            "sigma_0=f"   => \$sigma_0,
            "EM_iter=i" => \$num_iter,
            "Quasi_iter=i" => \$num_quasi_iter,
            "data=s" => \$data,
            "output_file=s" => \$output_file)
  or die("Error in command line arguments\n");

open(my $output, '>', $output_file) or die "Could not open file '$output_file' $!";
$lambda     = $lambda ? $lambda:0.01;
$gamma  = $gamma ? $gamma : $num_labels**2;

our $num_docs = 0;
our @input_lines;
my %Vocabulary;
open( F, "$data" ) or die "Couldn't load the data $data $!";
while (<F>) {
	chomp;
	my $line = $_;
	push(@input_lines, $line);
	$num_docs++;
}
close(F);
$varphi = $varphi ? $varphi : 0.1 * $num_docs;
$sigma_0  = $sigma_0 ? $sigma_0 : 0.1 * $num_docs;

print $output "data: ";
print $output $data;
print $output "\n";

print $output "num_topics: ";
print $output $num_topics;
print $output "\n";

print $output "dim: ";
print $output $dim;
print $output "\n";

print $output "lambda: ";
print $output $lambda;
print $output "\n";

print $output "varphi: ";
print $output $varphi;
print $output "\n";

print $output "gamma: ";
print $output $gamma;
print $output "\n";

print $output "sigma_0: ";
print $output $sigma_0;
print $output "\n";

print $output "EM_iter: ";
print $output $num_iter;
print $output "\n";

print $output "Quasi_iter: ";
print $output $num_quasi_iter;
print $output "\n";

my %all_tokens;
my @doc_single_label;
my %doc_labels;
my %all_tokens_count;
my @doc_length;
my $c = 0;

foreach (@input_lines) {
	my $line = $_;
	my @tokens = split( /\s/, $line );
	$doc_length[$c] = @tokens - $num_labels;
	my @unique_tokens;
	my @count_unique_tokens;
	my $count = 0;
	my %hash_w;
	my @labels;
	$doc_num_labels[$c] = 0;
	for ( my $i = 0 ; $i < $num_labels ; $i++ ) {
		$labels[$i] = $tokens[$i];
		if($labels[$i]!=0){
			$doc_num_labels[$c] = $doc_num_labels[$c] + 1;
		}
		if($i<$num_labels-1 && $labels[$i]==1){
			$doc_single_label[$c] = $i;
		}
	}
	for ( my $i = $num_labels ; $i < @tokens ; $i++ ) {
		if(!defined $Vocabulary{$tokens[$i]}) {
			$Vocabulary{$tokens[$i]} = 1;
		}
		if(defined $hash_w{$tokens[$i]}) {
			$hash_w{$tokens[$i]} = $hash_w{$tokens[$i]} + 1;
		}
		else{
			$hash_w{$tokens[$i]} = 1;
		}
	}
	foreach my $name (keys %hash_w) {
   		$unique_tokens[$count] = $name;
   		$count_unique_tokens[$count] = $hash_w{$name};
   		$count++;
	}
	$all_tokens{$c} = \@unique_tokens;
	$all_tokens_count{$c} = \@count_unique_tokens;
	$doc_labels{$c} = \@labels;
	$c++;
}

our $wordsize = (keys %Vocabulary);
our %beta;
our %phi;
our %xai;
our %mu;

our %update_beta_numerator;
our @update_beta_denominator;

print $output "initialize phi \n";
for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
	my @position;
	for ( my $d = 0 ; $d < $dim ; $d++ ) {
		$position[$d]  = &gaussian_rand * 0.01;
		print $output $position[$d];
		print $output "\n";
	}
	print $output "---------------\n";
	$phi{$i}     = \@position;
}

print $output "initialize mu \n";
for ( my $i = 0 ; $i < $num_labels ; $i++ ) {
	my @position;
	for ( my $d = 0 ; $d < $dim ; $d++ ) {
		$position[$d]  = &gaussian_rand * 0.01;
		print $output $position[$d];
		print $output "\n";
	}
	print $output "---------------\n";
	$mu{$i}     = \@position;
}


print $output "initialize xai \n";
for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
	my @position;
	for ( my $d = 0 ; $d < $dim ; $d++ ) {
		$position[$d]  = &gaussian_rand * 0.01;
		print $output $position[$d];
		print $output "\n";
	}
	print $output "---------------\n";
	$xai{$i}     = \@position;
}

for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
	my @words;
	my $denominator = 0;
	for ( my $j = 0 ; $j < $wordsize ; $j++ ) {
		$words[$j] = -log( 1.0 - rand );
		$denominator += $words[$j];
	}
	for ( my $j = 0 ; $j < $wordsize ; $j++ ) {
		$words[$j] = $words[$j] / $denominator;
	}
	$beta{$i} = \@words;
}

our %pr_lgx;
our %pr_zglx;
our %sum_lzgnm_overw;
our %sum_zgnm_overw;
our %sum_lgnm_overw;

my $quasi = Algorithm::LBFGS->new;
$quasi->set_param(max_iterations => $num_quasi_iter);

my $eval_short = sub {
	my $x = shift;
	my $sum   = 0;
	my $docid = 0;
	my %grad;
	my %grad_mu;
	my @gradient;
	my @numerators;
	my $offset_index_label = ($num_topics+$num_docs)*$dim;
	for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
		my @arr;
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			my $index = $j * $dim + $k;
			$arr[$k] = - $varphi * $x->[$index];
			$gradient[$index] = 0;
		}
		$grad{$j} = \@arr;
	}
	for ( my $j = 0 ; $j < $num_labels ; $j++ ) {
		my @arr;
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			my $index = $offset_index_label + $j * $dim + $k;
			$arr[$k] = - $sigma_0 * $x->[$index];
		}
		$grad_mu{$j} = \@arr;
	}
	
	for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
		my @prs;
		my $denominator = 0;
		my $offset_doc = $num_topics * $dim + $i*$dim;
		
		for ( my $j = 0 ; $j < $num_labels ; $j++ ) {
			my $d = 0;
			my $offset_label = $offset_index_label + $j * $dim;
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				$d += ($x->[$offset_doc + $k] - $x->[$offset_label + $k])**2;
			}
			$numerators[$j] = exp((-0.5) * $d);
			$denominator += $numerators[$j];
		}
		for ( my $j = 0 ; $j < $num_labels ; $j++ ) {
			$prs[$j] = $numerators[$j]/$denominator;
		}
		$pr_lgx{$i} = \@prs;
		
		my %pr;
		for ( my $j = 0 ; $j < $num_labels ; $j++ ) {
			my @prs;
			my $denominator = 0;
			my $offset_label = $offset_index_label + $j * $dim;
			for ( my $l = 0 ; $l < $num_topics ; $l++ ) {
				my $d = 0;
				my $offset_topic = $l * $dim;
				for ( my $k = 0 ; $k < $dim ; $k++ ) {
					$d += ($x->[$offset_doc + $k] - $x->[$offset_topic + $k])**2;
					$d += ($x->[$offset_label + $k] - $x->[$offset_topic + $k])**2;
				}
				$numerators[$l] = exp((-0.5) * $d);
				$denominator += $numerators[$l];
			}
			for ( my $l = 0 ; $l < $num_topics ; $l++ ) {
				$prs[$l] = $numerators[$l] / $denominator;
			}
			$pr{$j} = \@prs;
		}
		$pr_zglx{$i} = \%pr;
						
		$docid = $i;
		my @tokens = @{$all_tokens{$docid}};
		my @count_tokens = @{$all_tokens_count{$docid}};
		 
		my $p_lgx  = $pr_lgx{$docid};
		
		my $temp = 0;
		for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
			if($doc_labels{$i}->[$l]==1){
				$temp += $p_lgx->[$l];
			}
		}
		$sum += $doc_length[$i] * log($temp);
		
		my $t = 0;
		for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
			for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
				$t += $sum_lzgnm_overw{$docid}->{$l}->[$j] * log($p_lgx->[$l] * $pr_zglx{$i}->{$l}->[$j]);
			}
		}
		$sum += $t;
		
		my $length = 0;
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$length += ($x->[$offset_doc + $k]-$x->[$offset_index_label + $doc_single_label[$docid] * $dim + $k])**2;
		}
		$sum += (-$gamma/2) * $length;
		
		my @gra;
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$gra[$k] = 0;
		}
		#gradient wrt phi
		for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
			my $offset_topic = $j * $dim;
			for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
				my $offset_label = $offset_index_label + $l*$dim;
				my $p_zglx = $pr_zglx{$i}->{$l}->[$j];
				for ( my $k = 0 ; $k < $dim ; $k++ ) {
					my $t;
					$t = ($sum_lgnm_overw{$docid}->[$l] * $p_zglx - $sum_lzgnm_overw{$docid}->{$l}->[$j] ) * ( 2 * $x->[$offset_topic + $k] - $x->[$offset_label + $k] - $x->[$offset_doc + $k] );
					$grad{$j}->[$k] +=  $t;
				}
			}
		}
		
		#gradient wrt xai
		$temp = 0;
		for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
			my $offset_label = $offset_index_label + $l*$dim;
			my $p_lgx = $pr_lgx{$i}->[$l];
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				$gra[$k] += $doc_length[$i]*$p_lgx * ($x->[$offset_doc + $k] - $x->[$offset_label + $k]);
			}
			if($doc_labels{$i}->[$l]==1){
				$temp += $p_lgx;
			}
		}
		for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
			if($doc_labels{$i}->[$l]==1){
				my $offset_label = $offset_index_label + $l*$dim;
				my $p_lgx = $pr_lgx{$i}->[$l];
				for ( my $k = 0 ; $k < $dim ; $k++ ) {
					$gra[$k] += -$doc_length[$i]*($x->[$offset_doc + $k] - $x->[$offset_label + $k])*$p_lgx/$temp;
				}
			}
		}
		
		for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
			my $offset_label = $offset_index_label + $l*$dim;
			my $p_lgx = $pr_lgx{$i}->[$l];
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				my $t;
				$t = ($doc_length[$docid] * $p_lgx - $sum_lgnm_overw{$docid}->[$l]) * ($x->[$offset_doc + $k] - $x->[$offset_label + $k]);
				$gra[$k] +=  $t;
			}
			for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
				my $offset_topic = $j * $dim;
				for ( my $k = 0 ; $k < $dim ; $k++ ) {
					$gra[$k] += ($sum_lgnm_overw{$docid}->[$l] * $pr_zglx{$i}->{$l}->[$j] - $sum_lzgnm_overw{$docid}->{$l}->[$j])* ( $x->[$offset_doc + $k] - $x->[$offset_topic + $k]);					
				}
			}
		}
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$gradient[$offset_doc + $k] = -($gra[$k] - $gamma * ($x->[$offset_doc + $k]-$x->[$offset_index_label + $doc_single_label[$docid] * $dim + $k]));
		}
		#gradient wrt mu
		for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
			my $p_lgx = $pr_lgx{$i}->[$l];
			my $offset_label = $offset_index_label + $l*$dim;
			if($doc_labels{$i}->[$l]==0){
				for ( my $k = 0 ; $k < $dim ; $k++ ) {
					$grad_mu{$l}->[$k] += -$doc_length[$i]*$p_lgx*($x->[$offset_doc + $k] - $x->[$offset_label + $k]);
				}
			}
			else{
				for ( my $k = 0 ; $k < $dim ; $k++ ) {
					$grad_mu{$l}->[$k] += $doc_length[$i]*($p_lgx/$temp - $p_lgx)*($x->[$offset_doc + $k] - $x->[$offset_label + $k]);
				}
			}
		}

		for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
			my $p_lgx = $pr_lgx{$i}->[$l];
			my $offset_label = $offset_index_label + $l*$dim;
				
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				$grad_mu{$l}->[$k] += ($sum_lgnm_overw{$docid}->[$l] - $doc_length[$docid] * $p_lgx)*($x->[$offset_doc + $k] - $x->[$offset_label + $k]); 
			}
			
			for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
				my $p_zglx = $pr_zglx{$i}->{$l}->[$j];
				for ( my $k = 0 ; $k < $dim ; $k++ ) {
					$grad_mu{$l}->[$k] += ($sum_lgnm_overw{$docid}->[$l]* $p_zglx - $sum_lzgnm_overw{$docid}->{$l}->[$j])*($x->[$offset_label + $k] - $x->[$j * $dim + $k]); 
				}
			}
			
			if($doc_single_label[$docid]==$l){
				for ( my $k = 0 ; $k < $dim ; $k++ ){
					$grad_mu{$l}->[$k] += $gamma * ($x->[$offset_doc + $k]-$x->[$offset_index_label + $doc_single_label[$docid] * $dim + $k]);
				}
			}
		}
	}
	
	for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
		my $length = 0;
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$length += $x->[$j*$dim + $k]**2;
		}
		$sum += (-$varphi/2) * $length;
	}
	
	for ( my $j = 0 ; $j < $num_labels ; $j++ ) {
		my $length = 0;
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$length += $x->[$offset_index_label + $j*$dim + $k]**2;
		}
		$sum += (-$sigma_0/2) * $length;
	}
	
	my $f = -$sum;
	print ($f);
	print ("\n");
	for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$gradient[$j * $dim + $k] = -$grad{$j}->[$k];
		}
	}
	for ( my $j = 0 ; $j < $num_labels ; $j++ ) {
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$gradient[$offset_index_label + $j*$dim + $k] = -$grad_mu{$j}->[$k];
		}
	}
	return ( $f, \@gradient );
};
#EM
for ( my $i = 0 ; $i < $num_iter ; $i++ ) {
	print($i);
	print("\n");
	
	print $output ($i);
	print $output ("\t");
	&estep();
	&mstep();
	my $likelihood = 0;
	for ( my $d = 0 ; $d < $num_docs ; $d++ ) {
		my $p_lgx  = $pr_lgx{$d};
		
		my $temp = 0;
		for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
			if($doc_labels{$d}->[$l]==1){
				$temp += $p_lgx->[$l];
			}
		}
		$likelihood += $doc_length[$d] * log($temp);
	}
	for ( my $d = 0 ; $d < $num_docs ; $d++ ) {
		my @tokens = @{$all_tokens{$d}};
		my @count_tokens = @{$all_tokens_count{$d}};
		
		my $p_lgx  = $pr_lgx{$d};
		
		for ( my $m = 0 ; $m < @tokens ; $m++ ) {
			my $t = 0;
			for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
				for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
					$t += ($p_lgx->[$l] * $pr_zglx{$d}->{$l}->[$j] * $beta{$j}->[$tokens[$m]]);
				}
			}
			$likelihood += $count_tokens[$m] * log($t);
		}
	}
	$likelihoods[$i] = $likelihood;
	
	#print coordinates at each iteration
	if(($i+1)%10==0){
		print $output "\nCoordinates at Iteration ";
		print $output $i;
		print $output "\n";
		print $output "Topics: \n";
		for ( my $t = 0 ; $t < $num_topics ; $t++ ) {
			for ( my $d = 0 ; $d < $dim ; $d++ ) {
				print $output $phi{$t}[$d];
				print $output "\t";
			}
			print $output "\n";
		}
		print $output "Documents: \n";
		for ( my $t = 0 ; $t < $num_docs ; $t++ ) {
			for ( my $d = 0 ; $d < $dim ; $d++ ) {
				print $output $xai{$t}[$d];
				print $output "\t";
			}
			print $output "\n";
		}
		print $output "Labels: \n";
		for ( my $t = 0 ; $t < $num_labels ; $t++ ) {
			for ( my $d = 0 ; $d < $dim ; $d++ ) {
				print $output $mu{$t}[$d];
				print $output "\t";
			}
			print $output "\n";
		}
	}
}
print $output ("Likelihood\n");
for ( my $i = 0 ; $i < $num_iter ; $i++ ) {
	print $output ($likelihoods[$i]);
	print $output ("\n");
}

# output to file
print $output "\n-------Xai-------\n";
for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
	for ( my $d = 0 ; $d < $dim ; $d++ ) {
		if($d==0){
			print $output $xai{$i}[$d];
		}
		else{
			print $output "\t";
			print $output $xai{$i}[$d];
		}
	}
	print $output "\n";
}

print $output "\n-------Phi-------\n";
for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
	for ( my $d = 0 ; $d < $dim ; $d++ ) {
		if($d==0){
			print $output $phi{$i}[$d];
		}
		else{
			print $output "\t";
			print $output $phi{$i}[$d];
		}
	}
	print $output "\n";
}

print $output "\n-------Mu-------\n";
for ( my $i = 0 ; $i < $num_labels ; $i++ ) {
	for ( my $d = 0 ; $d < $dim ; $d++ ) {
		if($d==0){
			print $output $mu{$i}[$d];
		}
		else{
			print $output "\t";
			print $output $mu{$i}[$d];
		}
	}
	print $output "\n";
}
print $output "\n-------Verbose-------\n";
my $docid = 0;
print $output "lgivend \n";
foreach (@input_lines) {
	my $p_lgx = $pr_lgx{$docid};
	for ( my $j = 0 ; $j < $num_labels ; $j++ ) {
		print $output $p_lgx->[$j];
		print $output "\n";
	}
	print $output "-------------\n";
	$docid++;
}
print $output "zgivenlx \n";
for(my $d = 0;$d <$num_docs; $d++){
	for ( my $i = 0 ; $i < $num_labels ; $i++ ) {
		my $p_zglx = $pr_zglx{$d}->{$i};
		for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
			print $output $p_zglx->[$j];
			print $output "\n";
		}
		print $output "-------------\n";
	}	
}

print $output "Phi\n";
print $output Dumper(%phi);
print $output "Xai\n";
print $output Dumper(%xai);
print $output "Mu\n";
print $output Dumper(%mu);
print $output "beta\n";
print $output Dumper(%beta);

sub update_beta {
	my $topic = shift;
	for ( my $j = 0 ; $j < $wordsize ; $j++ ) {
		$beta{$topic}->[$j] =
		  ($update_beta_numerator{$topic}->[$j] + $lambda ) /
		  ($update_beta_denominator[$topic] + $lambda * $wordsize);
	}
};

sub update_beta_denominator {
	my @result;

	for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
		my $denominator = 0;
		for ( my $docid = 0 ; $docid < $num_docs ; $docid++ ) {
			$denominator += $sum_zgnm_overw{$docid}->[$j];
		}
		push( @result, $denominator );
	}
	return @result;
};

sub mstep {
	@update_beta_denominator = update_beta_denominator();
	for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
		&update_beta($i);
	}
	#using quasi newton
	my @x0;
	for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
		for ( my $d = 0 ; $d < $dim ; $d++ ) {
			$x0[$i * $dim + $d] = $phi{$i}[$d];
		}
	}
	my $offset = $num_topics * $dim;
	for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
		for ( my $d = 0 ; $d < $dim ; $d++ ) {
			$x0[$offset + $i * $dim + $d] = $xai{$i}[$d];
		}
	}
	$offset = $offset + $num_docs * $dim;
	for ( my $i = 0 ; $i < $num_labels ; $i++ ) {
		for ( my $d = 0 ; $d < $dim ; $d++ ) {
			$x0[$offset + $i * $dim + $d] = $mu{$i}[$d];
		}
	}
	print("start quasi\n");
	my $x;
	$x = $quasi->fmin($eval_short, \@x0);
	print("end quasi\n");
	for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
		for ( my $d = 0 ; $d < $dim ; $d++ ) {
			$phi{$i}[$d] = $x->[$i * $dim + $d];
		}
	}
	$offset = $num_topics * $dim;
	for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
		for ( my $d = 0 ; $d < $dim ; $d++ ) {
			$xai{$i}[$d] = $x->[$offset + $i * $dim + $d];
		}
	}
	$offset = $offset + $num_docs * $dim;
	for ( my $i = 0 ; $i < $num_labels ; $i++ ) {
		for ( my $d = 0 ; $d < $dim ; $d++ ) {
			$mu{$i}[$d] = $x->[$offset + $i * $dim + $d];
		}
	}
	return;
};

sub estep {
	for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
		my @words;
		for ( my $j = 0 ; $j < $wordsize ; $j++ ) {
			$words[$j] = 0;
		}
		$update_beta_numerator{$i} = \@words;
	}
	for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
		my @prs;
		my $denominator = 0;
		my @numerators;
		
		for (my $j = 0 ; $j < $num_labels ; $j++) {
			my $euclid = 0;
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				$euclid += ($xai{$i}->[$k]-$mu{$j}->[$k])**2;
			}
			$numerators[$j] = exp((-0.5) * ($euclid));
			$denominator += $numerators[$j];
		}
		for ( my $j = 0 ; $j < $num_labels ; $j++ ) {
			my $pr = $numerators[$j]/$denominator;
			$prs[$j] = $pr;
		}
		$pr_lgx{$i} = \@prs;
	}
	
	
	for ( my $d = 0 ; $d < $num_docs ; $d++ ) {
		my %pr;
		for ( my $i = 0 ; $i < $num_labels ; $i++ ) {
			my @prs;
			my $denominator = 0;
			my @numerators;
			for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
				my $euclid = 0;
				for ( my $k = 0 ; $k < $dim ; $k++ ) {
					$euclid += ($xai{$d}->[$k]-$phi{$j}->[$k])**2;
					$euclid += ($mu{$i}->[$k]-$phi{$j}->[$k])**2;
				}
				$numerators[$j] = exp((-0.5) * $euclid);
				$denominator += $numerators[$j];
			}
			for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
				my $pr = $numerators[$j]/$denominator;
				$prs[$j] = $pr;
			}
			$pr{$i} = \@prs;
		}
		$pr_zglx{$d} = \%pr;
	}
	for ( my $docid = 0 ; $docid < $num_docs ; $docid++ ) {
		my @tokens = @{$all_tokens{$docid}};
		my @count_tokens = @{$all_tokens_count{$docid}};
		
		my @sum_zgnm;
		my @sum_lgnm;
		my %sum_lzgnm;
		
		for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
			$sum_zgnm[$j] = 0;
		}
		for ( my $j = 0 ; $j < $num_labels ; $j++ ) {
			$sum_lgnm[$j] = 0;
		}
		for ( my $j = 0 ; $j < $num_labels ; $j++ ) {
			my @sum;
			for ( my $t = 0 ; $t < $num_topics ; $t++ ) {
				$sum[$t] = 0;
			}
			$sum_lzgnm{$j} = \@sum;
		}
		my $p_lgx = $pr_lgx{$docid};
		
		for ( my $i = 0 ; $i < @tokens ; $i++ ) {
			my %prs_lz;
			my $denominator = 0;
			my %numerator;
			for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
				my $p_zglx = $pr_zglx{$docid}->{$l};
				my @numer;
				for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
					my $t = $p_lgx->[$l] * $p_zglx->[$j] * $beta{$j}->[$tokens[$i]];
					$denominator += $t;
					$numer[$j] = $t;
				}
				$numerator{$l} = \@numer;
			}
			
			for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
				my @pr_lz;
				my $sum = 0;
				for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
					my $p = $numerator{$l}->[$j]/$denominator;
					$sum += $p;
					$pr_lz[$j] = $p;
					$sum_lzgnm{$l}->[$j] = $sum_lzgnm{$l}->[$j] + $count_tokens[$i]*$p;
				}
				$prs_lz{$l} = \@pr_lz;
				$sum_lgnm[$l] += $sum *  $count_tokens[$i];
			}
			
			for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
				my $sum = 0;
				for ( my $l = 0 ; $l < $num_labels ; $l++ ) {
					$sum += $prs_lz{$l}->[$j];
				}
				my $t = $sum * $count_tokens[$i];
				$sum_zgnm[$j] += $t;
				$update_beta_numerator{$j}->[$tokens[$i]] += $t;
			}
		}
		$sum_lzgnm_overw{$docid} = \%sum_lzgnm;
		$sum_lgnm_overw{$docid} = \@sum_lgnm;
		$sum_zgnm_overw{$docid} = \@sum_zgnm;
	}
	return;
};


sub gaussian_rand {
	#This function is from Perl Cookbook, 2nd Edition (http://www.oreilly.com/catalog/perlckbk2/)
	
	my ( $u1, $u2 );    # uniformly distributed random numbers
	my $w;              # variance, then a weight
	my ( $g1, $g2 );    # gaussian-distributed numbers

	do {
		$u1 = 2 * rand() - 1;
		$u2 = 2 * rand() - 1;
		$w  = $u1 * $u1 + $u2 * $u2;
	} while ( $w >= 1 );

	$w  = sqrt( ( -2 * log($w) ) / $w );
	$g2 = $u1 * $w;
	$g1 = $u2 * $w;

	# return both if wanted, else just one
	return wantarray ? ( $g1, $g2 ) : $g1;
};
