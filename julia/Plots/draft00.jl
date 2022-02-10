using Plots

plot() #empty Plot object
plot(4) #initialize with 4 empty series
plot(rand(10)) #1 series... x = 1:10
plot(rand(10,5)) #5 series... x = 1:10
plot(rand(10), rand(10)) # 1 series
plot(rand(10,5), rand(10)) # 5 series... y is the same for all
plot(rand(10), rand(10,5)) # 5 series... x is the same for all
plot([sin,cos], 0:0.1:π) # 2 series, sin.(x) and cos.(x)
plot([sin,cos], 0, π) # sin and cos on the range [0, π]

xdata = 0:0.01:(2*π)
plot(xdata, sin.(xdata))
plot!(xdata, sin.(2*xdata))
plot!(title="New Title", xlabel="New xlabel", ylabel="New ylabel")
plot!(xlims=(-0.1,2*π+0.1), ylims=(-1.1,1.1), xticks=0:0.5:(2*π), yticks=[-0.8,-0.2,0.2,0.8])
# plot!(xaxis = ("mylabel", :log10, :flip))
# xaxis!("mylabel", :log10, :flip)

default(:size) #get default value
default(:legend)
# default(size=(600,400), leg=false)

# gui() #plotting at the REPL (without semicolon) implicityly calls gui()

x = 1:10
y = rand(10, 2)
z = rand(10)
p = plot(x, y)
plot!(p, x, z) #Plots.CURRENT_PLOT


x = 1:10
y = rand(10, 2)
p = plot(x, y, title="two lines", label=["line1" "line2"], linewidth=3) #lw=3
xlabel!("My x label")
# plot!(xlabel="My x label")
# xlabel!(p, "My x label")
# savefig("tbd00.png") #"tbd00.pdf" for vector graphic
# savefig(p, "tbd00.png")

#backend
# plotly()
# gr()


x = 1:10
y = rand(10, 2)
plot(x, y, seriestype=:scatter)
# scatter(x, y, title="plot with scatter()")


function hf0(key="233")
    x = 1:10
    y = rand(10, 2)
    display(scatter(x, y, title="hf0(x=$(key))"))
    println("aha")
end


x = 1:10
y = rand(10, 4)
# plot(x, y, layout=(4,1))
p1 = plot(x, y, legend=false)
p2 = scatter(x, y)
p3 = plot(x, y, xlabel="labelled plot", linewidth=3)
p4 = histogram(x, y)
plot(p1, p2, p3, p4, layout=(2,2))

closeall()
