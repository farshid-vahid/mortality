# Julia file for reproducing graphs and results in Guppy, Nguyen and Vahid
# Julia version 1.0.3 (2018-12-18)
# Official https://julialang.org/ release
# Changing the country code and gender on the first two lines will produce results for males and females in different countries
# Country codes are given in the paper
# All data needs to be in a directory named data in the current directory
country = "USA"
gender = "male"
vintage = "2019_08_09"
datafile = string("data/" , country  , "_" , gender , "_" , vintage , ".csv")
outfile = string(country , "_" , gender , "_forecasts.txt")
# add packages in case they are not installed, and call packages that will be used
Pkg.add("Plots")
Pkg.add("Distributions")
using Plots
using Statistics
using LinearAlgebra
using Distributions

# some functions
function df(yy,maxlag, determin,ic) # this function returns the adf statistic
  nyy=size(yy,1)
  dyy0=yy[2:nyy,:]-yy[1:nyy-1,:]
  yylag=yy[maxlag+1:nyy-1,:]
  dyy = dyy0[maxlag+1:nyy-1,:]
  ndyy = size(dyy,1)
  yks = ones(ndyy)
  if determin == 0
      X = yylag
  elseif determin == 1
      X = hcat(yylag,ones(ndyy))
  else
      X = hcat(yylag,ones(ndyy),1:ndyy)
  end
  XX = X
  ssr=[]
  for ii = 0:maxlag
      XX
    if ii>0
      XX=hcat(XX,dyy0[maxlag+1-ii:nyy-1-ii])
    end
    uhat=dyy-XX*inv(XX'XX)*XX'dyy
    ssr=vcat(ssr,uhat'uhat)
  end
  if ic==0  #use AIC
      icval = log.(ssr) + 2*(0:maxlag)
  elseif ic==1 #use HQ
      icval = log.(ssr) + 2*log(log(ndyy))*(0:maxlag)
  else #use BIC
      icval = log.(ssr) + log.(ndyy)*(0:maxlag)
  end
  nlag=argmin(icval[:,1])-1
  dyy=dyy0[nlag+1:nyy-1,:]
  ndyy=size(dyy,1)
  XX = yy[nlag+1:nyy-1,:]
  if determin==1
      XX = hcat(XX,ones(ndyy))
  elseif determin ==2
      XX = hcat(XX,ones(ndyy),1:ndyy)
  end
  for ii=1:nlag
    XX=hcat(XX,dyy0[nlag+1-ii:nyy-1-ii])
  end
  # β = inv(X'X)*X'dy    	#cool, can use greek letters - could use X\dy
  β = XX \ dyy
  uhat = dyy - XX * β
  s2 = uhat'uhat/(ndyy-size(XX,2))
  V = s2.*inv(XX'XX)
  t = β ./ sqrt.(diag(V))
  return t[1]
end

function dm(e0,e1,loss,ww) # This function returns the p-value of the Diebold-Mariano test
    if loss==1
        d = abs.(e0)-abs.(e1)
    elseif loss==2
        d = e0.^2 - e1.^2
    end
    nnn=size(d,1)
    vd=d'd/nnn
    for jjj=1:ww-1
        vd = vd + 2*(nnn-jjj)*d[jjj+1:nnn,1]'d[1:nnn-jjj,1]/(nnn^2)
    end
    dmstat = sqrt(nnn)*mean(d)/sqrt(vd)
    return(cdf(Normal(),dmstat))
end

# read data
using DelimitedFiles
mratedf = readdlm(datafile,',',skipstart=1)
mrate = mratedf[:,2:size(mratedf,2)]
lmr = log.(mrate)
n = size(lmr,1)
age = 50:89
year = 1950:(1950+n-1)


using Plots
test = plot(year,lmr,legend=false)
savefig(test,string(country,"_",gender,"lmr"))

inend=n-10
lmrin=lmr[1:inend,:]
dlmrin=lmrin[2:inend,:]-lmrin[1:inend-1,:]

using Statistics
ax = mean(lmrin, dims=1)
lmrinc=lmrin.-ax

using LinearAlgebra
F = svd(lmrinc)
bx = F.V[:,1]./sum(F.V[:,1])
kt = sum(F.V[:,1])*F.S[1,1].*F.U[:,1]
nn = size(kt,1)
dkt = kt[2:nn]-kt[1:nn-1]

lcdrifts = mean(dkt)*bx
rwddrifts = mean(dlmrin,dims=1)
test = plot(age,hcat(lcdrifts,rwddrifts'),label=["LC drift","RWD drift"], legend=:topleft)
savefig(test,string(country,"_",gender,"_drifts"))

adfcalc = zeros(40,1)
for i=1:40
    adfcalc[i,1]=df(lmrin[:,i],4,2,2)
end
adfcv = -3.491*ones(40,1)
bar(age,adfcalc,legend=false,grid=false,color=:red)
test = plot!(age,adfcv,color=:black,plot_title="ADF statistics for log mortality rates")
savefig(test,string(country,"_",gender,"_adf"))

# Compute Lee-Carter residuals and test if they are stationary
indep = hcat(ones(nn,1),kt)
axbx = indep \ lmrin
ehat = lmrin - indep*axbx
adfresid = zeros(40,1)
for i=1:40
    adfresid[i,1]=df(ehat[:,i],4,0,2)
end
mac2010 = -3.443*ones(40,1)
bar(age,adfresid,legend=false,grid=false,color=:red, plot_title="ADF statistic for Lee-Carter residuals")
test = plot!(age,mac2010,color=:black)
savefig(test,string(country,"_",gender,"_adfresid"))

# Compute the first order serial correlation in the residuals of the Lee-Carter model and plot them
rhos = zeros(40,1)
for i=1:40
    et=ehat[2:nn,i]
    elag = ehat[1:nn-1,i]
    rhos[i,1]=cor(et,elag)
end
stdrho=ones(40,1)*2/sqrt(nn)
bar(age,rhos,legend=false,color=:red, plot_title="First order autocorrelation of ADF residuals")
test = plot!(age,stdrho,color=:black)
savefig(test,string(country,"_",gender,"_rhoLCresid"))

# The sparse VECM
nin = size(dlmrin,1)
asparse = zeros(40,1)
bsparse = zeros(40,40)
residsparse = zeros(nin,40)
asparse[1,1]=mean(dlmrin[:,1])
residsparse[:,1]=dlmrin[:,1].-mean(dlmrin[:,1])
lmrinlag=lmrin[1:inend-1,:]
for i=2:40
    indep=hcat(ones(nin,1),(lmrinlag[:,i-1]-lmrinlag[:,i]))
    bhat = indep \ dlmrin[:,i]
    asparse[i,1]=bhat[1,1]
    bsparse[i,i-1]=bhat[2,1]
    bsparse[i,i]=-bhat[2,1]
    residsparse[:,i]=dlmrin[:,i]-indep*bhat
end

# Compute and plot the ADF statstic for the spreads
adfcv=-3.491*ones(40,1)
adfcalcsp=zeros(40,1)
for i=2:40
    adfcalcsp[i,1]=df((lmrin[:,i]-lmrin[:,i-1]),4,2,2)
end
bar(age,adfcalcsp,legend=false,grid=false,color=:red)
test = plot!(age,adfcv,color=:black,plot_title="ADF statistics for spreads")
savefig(test,string(country,"_",gender,"_adfSpreads"))

# Compute and plot the first order autocorrelation of the residuals of the SVECM
rhosparse=zeros(40,1)
for i=1:40
    rhosparse[i,1]=cor(residsparse[2:nin,i],residsparse[1:nin-1,i])
end
stdrho=ones(40,1)*2/sqrt(nin)
stdrho=hcat(stdrho,-stdrho)
bar(age,rhosparse,legend=false,color=:red)
test = plot!(age,stdrho,color=:black)
savefig(test,string(country,"_",gender,"_rhoVECMresid"))

# Forecasting: Three different loops producing 1-step, 5-step and 10-step ahead _forecasts
# This could perhaps written more efficiently, but it does the job

# Initialise matrices to store forecast errors
# Forecast errors will be appendend to these as we add an observation and repeat the forecasting
flcerr=Array{Float64}(undef, 0, 40)
flccerr=Array{Float64}(undef, 0, 40)
frwderr=Array{Float64}(undef, 0, 40)
fecmerr=Array{Float64}(undef, 0, 40)

# Start 10 period before the end, forecast one-step ahead, then add one observation and repeat
for lastobs=n-10:n-1
    lmrin=lmr[1:lastobs,:]
    dlmrin=lmrin[2:lastobs,:]-lmrin[1:lastobs-1,:]
    #Estimating Lee-Carter and RWD
    ax = mean(lmrin, dims=1)
    lmrinc=lmrin.-ax

    using LinearAlgebra
    F = svd(lmrinc)
    bx = F.V[:,1]./sum(F.V[:,1])
    kt = sum(F.V[:,1])*F.S[1,1].*F.U[:,1]
    nn = size(kt,1)
    dkt = kt[2:nn]-kt[1:nn-1]

    lcdrifts = mean(dkt)*bx
    rwddrifts = mean(dlmrin,dims=1)

    #Estimating the sparse VECM
    nin = size(dlmrin,1)
    asparse = zeros(40,1)
    bsparse = zeros(40,40)
    residsparse = zeros(nin,40)
    asparse[1,1]=mean(dlmrin[:,1])
    residsparse[:,1]=dlmrin[:,1].-mean(dlmrin[:,1])
    lmrinlag=lmrin[1:lastobs-1,:]
    for i=2:40
        indep=hcat(ones(nin,1),(lmrinlag[:,i-1]-lmrinlag[:,i]))
        bhat = indep \ dlmrin[:,i]
        asparse[i,1]=bhat[1,1]
        bsparse[i,i-1]=bhat[2,1]
        bsparse[i,i]=-bhat[2,1]
        residsparse[:,i]=dlmrin[:,i]-indep*bhat
    end
    #Forecast out of sample
    lmrout = lmr[lastobs+1:lastobs+1,:] # Actual values for the next year, funny indexing is to maintain type as a row vector
    lmroutlag = lmr[lastobs:lastobs,:]' # use this 40x1 vector to forecast from VECM recursively
    flc = ax + bx'*kt[nn,1] + lcdrifts'
    flcc = lmr[lastobs:lastobs,:]+lcdrifts'
    frwd = lmr[lastobs:lastobs,:]+rwddrifts
    lmroutlag = lmroutlag + asparse + bsparse*lmroutlag
    fecm = lmroutlag'
    # Collect forecast errors
    global flcerr = vcat(flcerr,(lmrout-flc))
    global flccerr = vcat(flccerr,(lmrout-flcc))
    global frwderr = vcat(frwderr,(lmrout-frwd))
    global fecmerr = vcat(fecmerr,(lmrout-fecm))
end

mae = 100*hcat(mean(abs.(flcerr)),mean(abs.(flccerr)),mean(abs.(frwderr)),mean(abs.(fecmerr)))
mse = hcat(mean(flcerr.^2),mean(flccerr.^2),mean(frwderr.^2),mean(fecmerr.^2))
rmse =100*sqrt.(mse)
using Printf
d = open(outfile,"w")
@printf(d, "%s \n", string("1-step ahead forecast evaluation results for ", country, " - ", gender))
@printf(d, "%s \n", "------------------------------------------------------------")

@printf(d, "%s \n\n", "MAPE for all ages from 50 to 89")

@printf(d, "%8.5s %8.5s %8.5s %8.5s \n", "LC", "LM", "RWD", "SpVECM")
for i=1:4
    @printf(d, "%8.4f", mae[1,i])
end
@printf(d,"\n\n")
@printf(d, "%s \n\n", "RMSE for all ages from 50 to 89")

@printf(d, "%8.5s %8.5s %8.5s %8.5s \n", "LC", "LM", "RWD", "SpVECM")
for i=1:4
    @printf(d, "%8.4f", rmse[1,i])
end
@printf(d,"\n\n")
@printf(d,"****************************************\n")
@printf(d,"Diebold-Mariano p-values for MAE loss\n")
@printf(d,"H0: sparse VECM is equivalent to model X\n")
@printf(d,"H1: sparse VECM is better than model X\n")
@printf(d,"%s %f \n", "Model X: Lee-Carter", dm(vec(fecmerr),vec(flcerr),1,5))
@printf(d,"%s %f \n", "Model X: Lee-Miller", dm(vec(fecmerr),vec(flccerr),1,5))
@printf(d,"%s %f \n", "Model X: RWD", dm(vec(fecmerr),vec(frwderr),1,5))
@printf(d,"Diebold-Mariano p-values for MSE loss\n")
@printf(d,"%s %f \n", "Model X: Lee-Carter", dm(vec(fecmerr),vec(flcerr),2,5))
@printf(d,"%s %f \n", "Model X: Lee-Miller", dm(vec(fecmerr),vec(flccerr),2,5))
@printf(d,"%s %f \n", "Model X: RWD", dm(vec(fecmerr),vec(frwderr),2,5))
@printf(d,"\n==============================================================\n")
@printf(d,"\n")

# Repeat loop for 5-step ahead forecasts
flcerr=Array{Float64}(undef, 0, 40)
flccerr=Array{Float64}(undef, 0, 40)
frwderr=Array{Float64}(undef, 0, 40)
fecmerr=Array{Float64}(undef, 0, 40)

flcerr5=Array{Float64}(undef, 0, 40)
flccerr5=Array{Float64}(undef, 0, 40)
frwderr5=Array{Float64}(undef, 0, 40)
fecmerr5=Array{Float64}(undef, 0, 40)

for lastobs=n-10:n-5
    lmrin=lmr[1:lastobs,:]
    dlmrin=lmrin[2:lastobs,:]-lmrin[1:lastobs-1,:]
    #Estimating Lee-Carter and RWD
    ax = mean(lmrin, dims=1)
    lmrinc=lmrin.-ax

    using LinearAlgebra
    F = svd(lmrinc)
    bx = F.V[:,1]./sum(F.V[:,1])
    kt = sum(F.V[:,1])*F.S[1,1].*F.U[:,1]
    nn = size(kt,1)
    dkt = kt[2:nn]-kt[1:nn-1]

    lcdrifts = mean(dkt)*bx
    rwddrifts = mean(dlmrin,dims=1)

    #Estimating the sparse VECM
    nin = size(dlmrin,1)
    asparse = zeros(40,1)
    bsparse = zeros(40,40)
    residsparse = zeros(nin,40)
    asparse[1,1]=mean(dlmrin[:,1])
    residsparse[:,1]=dlmrin[:,1].-mean(dlmrin[:,1])
    lmrinlag=lmrin[1:lastobs-1,:]
    for i=2:40
        indep=hcat(ones(nin,1),(lmrinlag[:,i-1]-lmrinlag[:,i]))
        bhat = indep \ dlmrin[:,i]
        asparse[i,1]=bhat[1,1]
        bsparse[i,i-1]=bhat[2,1]
        bsparse[i,i]=-bhat[2,1]
        residsparse[:,i]=dlmrin[:,i]-indep*bhat
    end
    #Forecast out of sample
    lmrout = lmr[lastobs+1:lastobs+5,:] # Actual values for the next 5 years
    lmroutlag = lmr[lastobs:lastobs,:]' # use this 40x1 vector to forecast from VECM recursively
    flc = zeros(5,40)
    flcc = zeros(5,40)
    frwd = zeros(5,40)
    fecm = zeros(5,40)
    for h=1:5
        flc[h,:] = ax + bx'*kt[nn,1] + h*lcdrifts'
        flcc[h,:] = lmr[lastobs:lastobs,:]+ h*lcdrifts'
        frwd[h,:] = lmr[lastobs:lastobs,:]+ h*rwddrifts
        lmroutlag = lmroutlag + asparse + bsparse*lmroutlag
        fecm[h,:] = lmroutlag'
    end
    # Collect 1 to 5 step ahead forecast errors
    global flcerr = vcat(flcerr,(lmrout-flc))
    global flccerr = vcat(flccerr,(lmrout-flcc))
    global frwderr = vcat(frwderr,(lmrout-frwd))
    global fecmerr = vcat(fecmerr,(lmrout-fecm))
    # Collect only 5-step ahead forecast errors
    global flcerr5 = vcat(flcerr5,(lmrout[5:5,:]-flc[5:5,:]))
    global flccerr5 = vcat(flccerr5,(lmrout[5:5,:]-flcc[5:5,:]))
    global frwderr5 = vcat(frwderr5,(lmrout[5:5,:]-frwd[5:5,:]))
    global fecmerr5 = vcat(fecmerr5,(lmrout[5:5,:]-fecm[5:5,:]))
end

mae = 100*hcat(mean(abs.(flcerr)),mean(abs.(flccerr)),mean(abs.(frwderr)),mean(abs.(fecmerr)))
mse = hcat(mean(flcerr.^2),mean(flccerr.^2),mean(frwderr.^2),mean(fecmerr.^2))
rmse =100*sqrt.(mse)

mae5 = 100*hcat(mean(abs.(flcerr5)),mean(abs.(flccerr5)),mean(abs.(frwderr5)),mean(abs.(fecmerr5)))
mse5 = hcat(mean(flcerr5.^2),mean(flccerr5.^2),mean(frwderr5.^2),mean(fecmerr5.^2))
rmse5 =100*sqrt.(mse5)


@printf(d, "%s \n", string("5-step ahead forecast evaluation results for ", country, " - ", gender))
@printf(d, "%s \n", "------------------------------------------------------------")

@printf(d, "%s \n\n", "MAPE for 1 to 5 step ahead forecasts")

@printf(d, "%8.5s %8.5s %8.5s %8.5s \n", "LC", "LM", "RWD", "SpVECM")
for i=1:4
    @printf(d, "%8.4f", mae[1,i])
end
@printf(d,"\n\n")
@printf(d, "%s \n\n", "RMSE for 1 to 5 step ahead forecasts")

@printf(d, "%8.5s %8.5s %8.5s %8.5s \n", "LC", "LM", "RWD", "SpVECM")
for i=1:4
    @printf(d, "%8.4f", rmse[1,i])
end
@printf(d,"\n\n")

@printf(d, "%s \n\n", "MAPE for 5 step ahead forecasts only")

@printf(d, "%8.5s %8.5s %8.5s %8.5s \n", "LC", "LM", "RWD", "SpVECM")
for i=1:4
    @printf(d, "%8.4f", mae5[1,i])
end
@printf(d,"\n\n")
@printf(d, "%s \n\n", "RMSE for 5 step ahead forecasts only")

@printf(d, "%8.5s %8.5s %8.5s %8.5s \n", "LC", "LM", "RWD", "SpVECM")
for i=1:4
    @printf(d, "%8.4f", rmse5[1,i])
end
@printf(d,"\n\n")
@printf(d,"****************************************\n")
@printf(d,"Diebold-Mariano p-values for MAE loss, 1 to 5 steps ahead\n")
@printf(d,"H0: sparse VECM is equivalent to model X\n")
@printf(d,"H1: sparse VECM is better than model X\n")
@printf(d,"%s %f \n", "Model X: Lee-Carter", dm(vec(fecmerr),vec(flcerr),1,5))
@printf(d,"%s %f \n", "Model X: Lee-Miller", dm(vec(fecmerr),vec(flccerr),1,5))
@printf(d,"%s %f \n", "Model X: RWD", dm(vec(fecmerr),vec(frwderr),1,5))
@printf(d,"Diebold-Mariano p-values for MSE loss, 1 to 5 steps ahead\n")
@printf(d,"%s %f \n", "Model X: Lee-Carter", dm(vec(fecmerr),vec(flcerr),2,5))
@printf(d,"%s %f \n", "Model X: Lee-Miller", dm(vec(fecmerr),vec(flccerr),2,5))
@printf(d,"%s %f \n", "Model X: RWD", dm(vec(fecmerr),vec(frwderr),2,5))

@printf(d,"Diebold-Mariano p-values for MAE loss, 5 steps ahead only\n")
@printf(d,"%s %f \n", "Model X: Lee-Carter", dm(vec(fecmerr5),vec(flcerr5),1,5))
@printf(d,"%s %f \n", "Model X: Lee-Miller", dm(vec(fecmerr5),vec(flccerr5),1,5))
@printf(d,"%s %f \n", "Model X: RWD", dm(vec(fecmerr5),vec(frwderr5),1,5))
@printf(d,"Diebold-Mariano p-values for MSE loss, 5 steps ahead only\n")
@printf(d,"%s %f \n", "Model X: Lee-Carter", dm(vec(fecmerr5),vec(flcerr5),2,5))
@printf(d,"%s %f \n", "Model X: Lee-Miller", dm(vec(fecmerr5),vec(flccerr5),2,5))
@printf(d,"%s %f \n", "Model X: RWD", dm(vec(fecmerr5),vec(frwderr5),2,5))
@printf(d,"\n==============================================================\n")
@printf(d,"\n")


# Repeat the loop for 10-step ahead forecasts
flcerr=Array{Float64}(undef, 0, 40)
flccerr=Array{Float64}(undef, 0, 40)
frwderr=Array{Float64}(undef, 0, 40)
fecmerr=Array{Float64}(undef, 0, 40)

flcerr10=Array{Float64}(undef, 0, 40)
flccerr10=Array{Float64}(undef, 0, 40)
frwderr10=Array{Float64}(undef, 0, 40)
fecmerr10=Array{Float64}(undef, 0, 40)

for lastobs=n-10:n-10
    lmrin=lmr[1:lastobs,:]
    dlmrin=lmrin[2:lastobs,:]-lmrin[1:lastobs-1,:]
    #Estimating Lee-Carter and RWD
    ax = mean(lmrin, dims=1)
    lmrinc=lmrin.-ax

    using LinearAlgebra
    F = svd(lmrinc)
    bx = F.V[:,1]./sum(F.V[:,1])
    kt = sum(F.V[:,1])*F.S[1,1].*F.U[:,1]
    nn = size(kt,1)
    dkt = kt[2:nn]-kt[1:nn-1]

    lcdrifts = mean(dkt)*bx
    rwddrifts = mean(dlmrin,dims=1)

    #Estimating the sparse VECM
    nin = size(dlmrin,1)
    asparse = zeros(40,1)
    bsparse = zeros(40,40)
    residsparse = zeros(nin,40)
    asparse[1,1]=mean(dlmrin[:,1])
    residsparse[:,1]=dlmrin[:,1].-mean(dlmrin[:,1])
    lmrinlag=lmrin[1:lastobs-1,:]
    for i=2:40
        indep=hcat(ones(nin,1),(lmrinlag[:,i-1]-lmrinlag[:,i]))
        bhat = indep \ dlmrin[:,i]
        asparse[i,1]=bhat[1,1]
        bsparse[i,i-1]=bhat[2,1]
        bsparse[i,i]=-bhat[2,1]
        residsparse[:,i]=dlmrin[:,i]-indep*bhat
    end
    #Forecast out of sample
    lmrout = lmr[lastobs+1:lastobs+10,:] # Actual values for the next 5 years
    lmroutlag = lmr[lastobs:lastobs,:]' # use this 40x1 vector to forecast from VECM recursively
    flc = zeros(10,40)
    flcc = zeros(10,40)
    frwd = zeros(10,40)
    fecm = zeros(10,40)
    for h=1:10
        flc[h,:] = ax + bx'*kt[nn,1] + h*lcdrifts'
        flcc[h,:] = lmr[lastobs:lastobs,:]+ h*lcdrifts'
        frwd[h,:] = lmr[lastobs:lastobs,:]+ h*rwddrifts
        lmroutlag = lmroutlag + asparse + bsparse*lmroutlag
        fecm[h,:] = lmroutlag'
    end
    # Collect 1 to 5 step ahead forecast errors
    global flcerr = vcat(flcerr,(lmrout-flc))
    global flccerr = vcat(flccerr,(lmrout-flcc))
    global frwderr = vcat(frwderr,(lmrout-frwd))
    global fecmerr = vcat(fecmerr,(lmrout-fecm))
    # Collect only 5-step ahead forecast errors
    global flcerr10 = vcat(flcerr10,(lmrout[10:10,:]-flc[10:10,:]))
    global flccerr10 = vcat(flccerr10,(lmrout[10:10,:]-flcc[10:10,:]))
    global frwderr10 = vcat(frwderr10,(lmrout[10:10,:]-frwd[10:10,:]))
    global fecmerr10 = vcat(fecmerr10,(lmrout[10:10,:]-fecm[10:10,:]))
end

mae = 100*hcat(mean(abs.(flcerr)),mean(abs.(flccerr)),mean(abs.(frwderr)),mean(abs.(fecmerr)))
mse = hcat(mean(flcerr.^2),mean(flccerr.^2),mean(frwderr.^2),mean(fecmerr.^2))
rmse =100*sqrt.(mse)

mae10 = 100*hcat(mean(abs.(flcerr10)),mean(abs.(flccerr10)),mean(abs.(frwderr10)),mean(abs.(fecmerr10)))
mse10 = hcat(mean(flcerr10.^2),mean(flccerr10.^2),mean(frwderr10.^2),mean(fecmerr10.^2))
rmse10 =100*sqrt.(mse10)

using Printf

@printf(d, "%s \n", string("10-step ahead forecast evaluation results for ", country, " - ", gender))
@printf(d, "%s \n", "------------------------------------------------------------")

@printf(d, "%s \n\n", "MAPE for 1 to 10 step ahead forecasts")

@printf(d, "%8.5s %8.5s %8.5s %8.5s \n", "LC", "LM", "RWD", "SpVECM")
for i=1:4
    @printf(d, "%8.4f", mae[1,i])
end
@printf(d,"\n\n")
@printf(d, "%s \n\n", "RMSE for 1 to 10 step ahead forecasts")

@printf(d, "%8.5s %8.5s %8.5s %8.5s \n", "LC", "LM", "RWD", "SpVECM")
for i=1:4
    @printf(d, "%8.4f", rmse[1,i])
end
@printf(d,"\n\n")

@printf(d, "%s \n\n", "MAPE for 10 step ahead forecasts only")

@printf(d, "%8.5s %8.5s %8.5s %8.5s \n", "LC", "LM", "RWD", "SpVECM")
for i=1:4
    @printf(d, "%8.4f", mae10[1,i])
end
@printf(d,"\n\n")
@printf(d, "%s \n\n", "RMSE for 10 step ahead forecasts only")

@printf(d, "%8.5s %8.5s %8.5s %8.5s \n", "LC", "LM", "RWD", "SpVECM")
for i=1:4
    @printf(d, "%8.4f", rmse10[1,i])
end
@printf(d,"\n\n")
@printf(d,"\n")
@printf(d,"****************************************\n")
@printf(d,"Diebold-Mariano p-values for MAE loss, 1 to 10 steps ahead\n")
@printf(d,"H0: sparse VECM is equivalent to model X\n")
@printf(d,"H1: sparse VECM is better than model X\n")
@printf(d,"%s %f \n", "Model X: Lee-Carter", dm(vec(fecmerr),vec(flcerr),1,10))
@printf(d,"%s %f \n", "Model X: Lee-Miller", dm(vec(fecmerr),vec(flccerr),1,10))
@printf(d,"%s %f \n", "Model X: RWD", dm(vec(fecmerr),vec(frwderr),1,10))
@printf(d,"Diebold-Mariano p-values for MSE loss, 1 to 10 steps ahead\n")
@printf(d,"%s %f \n", "Model X: Lee-Carter", dm(vec(fecmerr),vec(flcerr),2,10))
@printf(d,"%s %f \n", "Model X: Lee-Miller", dm(vec(fecmerr),vec(flccerr),2,10))
@printf(d,"%s %f \n", "Model X: RWD", dm(vec(fecmerr),vec(frwderr),2,10))

@printf(d,"Diebold-Mariano p-values for MAE loss, 10 steps ahead only\n")
@printf(d,"%s %f \n", "Model X: Lee-Carter", dm(vec(fecmerr10),vec(flcerr10),1,10))
@printf(d,"%s %f \n", "Model X: Lee-Miller", dm(vec(fecmerr10),vec(flccerr10),1,10))
@printf(d,"%s %f \n", "Model X: RWD", dm(vec(fecmerr10),vec(frwderr10),1,10))
@printf(d,"Diebold-Mariano p-values for MSE loss, 10 steps ahead only\n")
@printf(d,"%s %f \n", "Model X: Lee-Carter", dm(vec(fecmerr10),vec(flcerr10),2,10))
@printf(d,"%s %f \n", "Model X: Lee-Miller", dm(vec(fecmerr10),vec(flccerr10),2,10))
@printf(d,"%s %f \n", "Model X: RWD", dm(vec(fecmerr10),vec(frwderr10),2,10))
@printf(d,"\n==============================================================\n")
@printf(d,"\n")

close(d)
