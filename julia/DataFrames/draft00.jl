using DataFrames, Random

# ENV["COLUMNS"]
# ENV["LINES"]

z0 = DataFrame(A=1:3, B=rand(3), C=randstring.([2,3,3]))

z0.A #z0.B z0.C #view
z0[!,:A] #z0[!,:B] z0[!,:C] #Symbol("A") #view
z0[!,1] #z0[!,2] z0[!,3]
z0[:,:A] #copy
z0[:,:A]==z0[!,:A] #true
z0[:,:A]===z0[!,:A] #false
z0[!,Not(:A)]

names(z0) #->Array{Symbol,1}, 1 is the dimension
size(z0) #->Tuple{Int64,Int64}

z0 = DataFrame()
z0.A = 1:3
z0.B = rand(3)

z0 = DataFrame(A=Int[], B=String[]) #C=Any[]
push!(z0, (1,"M"))
push!(z0, [2,"N"])
push!(z0, Dict(:A=>4, :B=>"G"))

z0 = DataFrame(A=[missing])

N0 = 500
z0 = DataFrame(A=1:N0, B=rand(N0), C=randstring.(rand(1:10,N0)))
# show(z0, allrows=true, allcols=true)
first(z0, 23)
last(z0, 23)
z0[1:3,:]
z0[[1,5,10],:]
z0[:,[:A,:B]] #z0[:,(:A,:B)] invalid
z0[z0.A .< 100, :]

# serialize
import Serialization
z0 = DataFrame(A=1:3, B=rand(3), C=randstring.([2,3,3]))
Serialization.serialize("tbd00.jls", z0) #TODO maybe a better suffix
z1 = Serialization.deserialize("tbd00.jls")
