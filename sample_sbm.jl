include("src/main.jl")

using DelimitedFiles

n = 1200


p_in = 0.1
p_out = 0.01#p_in*n^(-0.2)



k = 10
data_name = "sbm_1200"

clusters_sizes = zeros(Int64,k)

for x in 1:k
  clusters_sizes[x] = n/k #trunc(Int,n/(x+1)^1.1)
end
n = sum(clusters_sizes)
data = spzeros(n,n)

let start = 0

  for s in clusters_sizes

    for i in (start + 1):(start + s)
      for j in (i+1):(start+s)
        d_ij = rand(Bernoulli(p_in))
        if d_ij == 1
          data[i,j] = 1
        end
      end
      for j in (start+s+1):n
        d_ij = rand(Bernoulli(p_out))
        if d_ij == 1
          data[i,j] = 1
        end
      end
    end
    start += s
  end

end

#spy(data)

while true
  println()
  println("Save data in .jld ? [y/n]")
  continue_ = chomp(readline())
  if continue_ == "n"
    break
  end
  if continue_ == "y"
    I_,J_,Val_ = findnz(data)
    writedlm(string("data/",data_name,".txt"),hcat(I_,J_))
    break
  end
end

#=
# Overlap
for i in 301:500
  for j in (i+1):500
    d_ij = rand(Bernoulli(p_in))
    if d_ij == 1
      data[i,j] = 1
    end
  end
end
=#
