using PyCall
using JLD
#using GenericSVD
@pyimport numpy as np
@pyimport networkx as nx
@pyimport scipy.sparse.csgraph as csg
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport data_prep as dp
@pyimport load_dist as ld
@pyimport distortions as dis

setprecision(BigFloat, 1024)


function big_gemv!(A,x_in,x_temp)
    (n,n) = size(A)
    Threads.@threads for i=1:n
        x_out[i] = big(0.)
        for j=1:n
            x_out[i] += A[i,j]*x_in[j]
        end
    end
    Threads.@threads for i=1:n
       x_in[i] = x_out[i]
    end
end
function power_method(A,d,tol;T=200)
    tol=big(1e-1)^tol
    (n,n) = size(A)
    x_all = big.(qr(randn(n,d))[1])
    _eig  = zeros(BigFloat, d)
    x_temp = zeros(BigFloat,n)
    
    for j=1:d
        x = view(x_all,:,j)
        x /= norm(x)
        for t=1:T            
            #x = A*x
            big_gemv!(A,x,x_temp)
            if j > 1
                #x -= sum(x_all[:,1:(j-1)]*diagm(vec(x'x_all[:,1:(j-1)])),2)
                yy = vec(x'view(x_all, :,1:(j-1)))
                for k=1:(j-1)
                    x -= view(x_all,:,k)*yy[k]
                end
            end
            nx = norm(x)
            x /= nx
            cur_dist = abs(nx - _eig[j])
            if !isinf(cur_dist) &&  min(cur_dist, cur_dist/nx) < tol
                println("\t Done with eigenvalue $(j) at iteration $(t) at abs_tol=$(Float64(abs(nx - _eig[j]))) rel_tol=$(Float64(abs(nx - _eig[j])/nx))")
                break
            end
            if t % 500 == 0
                println("\t $(t) $(cur_dist)\n\t\t $(cur_dist/nx)")
            end
            _eig[j]    = nx
        end
        x_all[:,j] = x 
    end
    return (_eig, x_all)
end

function power_method_sign(A,r,tol;verbose=false, T=200)
    _d, _U    = power_method(A'A,r, tol;T=T)
    X         = _U'A*_U 
    _d_signed = vec(diag(X))
    if verbose
        print("Log Off Diagonals: $( Float64(log(vecnorm( X - diagm(_d_signed)))))")
    end
    return _d_signed, _U
end

# Get scaling factor tau
function get_emb_par(G, k, eps, weighted)
    n       = G[:order]();
    degrees = Dict(G[:degree]());
    cd      = collect(degrees);
    d_max   = maximum([cd[i][2] for i in 1:n])

    (nu, tau) = (0, 0)
    	
    beta    = big(pi)/(big(1.2)*d_max)
    v       = -2*k*log(tan(beta/2))
    
    for edge in G[:edges]()   
        (deg1, deg2) = (degrees[edge[1]], degrees[edge[2]])      
        alpha        = 2*big(pi)/(max(deg1,deg2))-2*beta
        len          = -big(2)*k*log(tan(alpha/2))
        w            = weighted ? edge[2]["weight"] : 1
        nu           = (len/w > nu) ? len/w : nu
        tau          = (1+eps)/eps*v > w*nu ? ((1+eps)/eps*v)/w : nu
    end
    return tau
end


function distortion(H1, H2)
    n,_ = size(H1)
    mc, me, avg, good = 0,0,0,0;
    for i=1:n
        for j=i+1:n
            if !isnan(H2[i,j]) && H2[i,j] != Inf && H2[i,j] != 0 && H1[i,j] != 0
                avg += max(H2[i,j]/H1[i,j], H1[i,j]/H2[i,j]);

                if H2[i,j]/H1[i,j] > me
                    me = H2[i,j]/H1[i,j]
                end

                if H1[i,j]/H2[i,j] > mc
                    mc = H1[i,j]/H2[i,j]
                end
                good += 1
            end
        end
    end

    avg/=(good);
    return (convert(Float64, mc*me), convert(Float64, avg), n*(n-1)/2-good)
end

# this is classical MDS
function mds(Z, k, n)
    o = ones(n,1)
    H = eye(n)-1/n*o*o'
    B = -1/2*H*Z*H
    B = 1/2*(B+B')       
       
    lambdasM, usM = power_method_sign(B,k,tol) 

    posE = 0
    while (posE < k && lambdasM[posE+1] > 0)
        posE+=1;
    end

    Xrec = usM[:,1:posE-1] * diagm(lambdasM[1:posE-1] .^ 0.5);    
    return Xrec', posE-1
end


# hMDS exact:
function h_mds(Z, k, n, tol)
    println("First e call")
    tic()
    eval, evec = power_method_sign(Z,1,tol)  
    lambda = eval[1]
    u = evec[:,1]
    println("lambda = $(convert(Float64,lambda))")

    toc()
    #EZ = eig(Z);
    #lambda = EZ[1][n];
    #u = EZ[2][:,n];    
    
    if (u[1] < 0)
        u = -u;
    end;

    b     = big(1) + sum(u)^2/(lambda*u'*u);
    alpha = b-sqrt(b^2-big(1));
    u_s   = u./(sum(u))*lambda*(big(1)-alpha);
    d     = (u_s+big(1))./(big(1)+alpha);
    dinv  = big(1)./d;
    v     = diagm(dinv)*(u_s.-alpha)./(big(1)+alpha);
    D     = big.(diagm(dinv));
    
    M = -(D * Z * D - ones(n) * v' - v * ones(n)')/2;
    M = (M + M')/2;
       
    # power method:
    println("Second e call")
    tic()
    lambdasM, usM = power_method_sign(M,k,tol) 
    posE = 1;
    while (lambdasM[posE] > 0 && posE<k)
        posE+=1;
    end

    Xrec = usM[:,1:posE-1] * diagm(lambdasM[1:posE-1] .^ 0.5);
    Xrec = Xrec';
    toc()
    
    # low precision:
    #EM = eig(M);    
    #lambdasM = EM[1][(n-k+1):n];
    #usM = EM[2][:,(n-k+1):n];

    # using SVD:
    #sv = svdfact(M)
    #A = diagm(sv[:S].^0.5)
    #Xrec = (sv[:U][:,1:k]*A[1:k,1:k])'
       
    return Xrec, posE-1
end

data_set = parse(Int32,(ARGS[1]))
k = parse(Int32, (ARGS[2]))
scale = parse(Float64, (ARGS[3]))
prec = parse(Int64, (ARGS[4]))
tol = parse(Int64, (ARGS[5]))

setprecision(BigFloat, prec)

#println("Scaling = $(convert(Float64,scale))");
#println(string("./dists/dist_mat",data_set,".p"))   

#H = ld.load_dist_mat(string("./dists/dist_mat",data_set,".p"));
G = dp.load_graph(data_set)
H = ld.get_dist_mat(G);
n,_ = size(H)

Xmds, dim_mds = mds(H, k, n)

Z = (cosh.(big.(H.*scale))-1)./2

println("Doing HMDS...")
tic()
Xrec, found_dimension = h_mds(Z, k, n, tol)

# save the recovered points:
save(string("Xrec_dataset_",data_set,"r=",k,"prec=",prec,"tol=",tol,".jld"), "H", H);
toc()

if found_dimension > 1
    println("Building recovered graph...")
    tic()
    Zrec = big.(zeros(n, n));
    for i = 1:n
        for j = 1:n
            Zrec[i,j] = norm(Xrec[:,i] - Xrec[:,j])^2 / ((1 - norm(Xrec[:,i])^2) * (1 - norm(Xrec[:,j])^2));
        end
    end
    toc()

    # the MDS distances:
    Zmds = zeros(n,n)
    for i = 1:n
        for j = 1:n
            Zmds[i,j] = norm(Xmds[:,i] - Xmds[:,j])
        end
    end

    
    println("Getting metrics")
    tic()
    Hrec = acosh.(1+2*Zrec)
    Hrec = convert(Array{Float64,2},Hrec)
    Hrec /= convert(Float64,scale)
    
    dist_max, dist, good = dis.distortion(H, Hrec, n, 2)
    println("Distortion avg/max, bad = $(convert(Float64,dist)), $(convert(Float64,dist_max)), $(convert(Float64,good))")  
    mapscore = dis.map_score(H, Hrec, n, 2) 
    println("MAP = $(mapscore)")   
    println("Dimension = $(found_dimension)")
    toc() 
    
    println("----------------MDS Results-----------------")
    dist_max, dist, bad = dis.distortion(H, Zmds, n, 2)
    println("MDS Distortion avg/max, bad = $(convert(Float64,dist)), $(convert(Float64,dist_max)), $(convert(Float64,bad))")  
    mapscore = dis.map_score(H, Zmds, n, 2)
    println("MAP = $(mapscore)")   
    println("Bad Dists = $(bad)")
    println("Dimension = $( dim_mds)") 
    
    
else
    println("Dimension = 1!")
end
