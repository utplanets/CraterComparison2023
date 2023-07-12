import numpy as np
import numba
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from config import *
from data import *

def filter_data(df, xlim=None, ylim=None,dlim=None,x="Long",y="Lat",d="`Diameter (km)`"):
    """A Query based filter to filter out a grid box of data.
    
    Returns a copy.
    """
    
    for k,v in {x:xlim,y:ylim,d:dlim}.items():
        if v is not None:
            df = df.query(f"{k} >= {v[0]} and {k} <= {v[1]}").copy()
    return df

@numba.jit
def iouD(my_x,my_y,my_d,xv,yv,dv, dx,dy,dd, radius=3391, method="Benedix"):
    """Calculates the IOU and distance to the best candidate crater
    my_x, my_y, my_d -> my crater
    xv, yv, dv -> arrays/list of candidate craters
    dx -> error in longitude allowed
    dy -> error in latitude allowed
    radius -> radius of planet
    method -> "Benedix" or "Lee" to choose the comparison method

    Returns:
        Crater with the maximum IOU (-IOU and index), the closest crater (distance and index).
    """

    #Find the maximum iou
    index_dist=-1
    index_iou=-1
    mind=1e9
    maxiou=0
    count=0
    k2d_y = radius * (2*np.pi)/360
    k2d_x = k2d_y * np.cos(my_y*np.pi/180)
    
    for i in range(len(xv)):
        iou=0
        check=False
        #Check X
        if method=="Benedix":
            if np.abs(my_d - dv[i]) < dd*my_d:
                check=True
            if check:
                if my_x < -180+dx:
                    if xv[i] > -180 and xv[i] < (my_x+dx):
                        check=True
                    elif xv[i] > 360+(my_x-dx) and xv[i] < 180:
                        check=True
                elif my_x > 180-dx:
                    if xv[i] > (my_x-dx) and xv[i] < 180:
                        check=True
                    elif xv[i] > -180 and xv[i] < -360+(my_x+dx):
                        check=True
                elif xv[i] >= (my_x-dx) and xv[i] <=my_x + dx:
                        check=True
            #check Y        
            if check:
                if yv[i] >= (my_y-dy) and yv[i] <=my_y + dy:
                    check=True
                else:
                    check=False
        elif method=="Lee":
            if np.abs(my_d - dv[i]) < dd*my_d:
                check=True
            if check:
                delta_x = xv[i]-my_x
                delta_y = yv[i]-my_y
                
                if delta_x > 180:
                    delta_x = 360 - delta_x
                if delta_x < -180:
                    delta_x = 360 + delta_x

                check = (np.abs(delta_x)*k2d_x < dx*my_d) and (np.abs(delta_y)*k2d_y < dy*my_d)

        if not check:
            iou=0
        else:
            count+=1
            #calc iou!
            x1 = my_x
            y1 = my_y
            r1 = my_d/2

            x2 = xv[i]
            y2 = yv[i]
            r2 = dv[i]/2

            #great_circle
            d2r = np.pi/180
            v = np.sin(y1*d2r) * np.sin(y2*d2r) + np.cos(y1*d2r)*np.cos(y2*d2r) * np.cos((x1-x2)*d2r)
            
            if v > (1 - 1e-10) and v<(1+1e-10):
                v=1
            elif v > -(1 - 1e-10) and v<-(1+1e-10):
                v=-1
            px1 = 0
            py1 = 0
            px2 = 0
            py2 = radius *np.arccos(v)
            p1 = np.array([px1,py1])
            p2 = np.array([px2,py2])
            if r1 < r2: #swap
                _ = r2
                r2 = r1
                r1 = _
                
            d = py2
            my_distance = d
            if d < mind:
                mind = d
                index_dist = i
            if d > r1 + r2:
                intersect = 0
                union = np.pi*r1**2 + np.pi*r2**2
                total = np.pi*r1**2 + np.pi*r2**2
                iou=intersect/union

            elif d <= r1-r2:
                intersect = np.pi*r2**2
                union = np.pi*r1**2
                total = np.pi*r1**2 + np.pi*r2**2
                iou=intersect/union
            elif d < r1+r2:
                a = (r1**2 - r2**2 + d**2)/(2*d)
                b = (r2**2 - r1**2 + d**2)/(2*d)
                if abs(r1-a)<1e-6:
                    h=0
                else:
                    h = np.sqrt(r1**2-a**2)
                p5 = p1+(a/d)*(p2-p1)
                v12 = p2-p1

                rc = np.array([[0,1.],[-1.,0]])
                v54 = (h/d)*np.dot(rc,v12)
                p4 = p5 +v54
                ra = np.array([[0,-1.],[1,0.]])
                v53 = (h/d)*np.dot(ra,v12)
                p3 = p5 + v53

                theta1 = 2*np.arccos(a/r1)
                s1 = (theta1-np.sin(theta1))/2 * r1**2
                theta2 = 2*np.arccos(b/r2)
                s2 = (theta2-np.sin(theta2))/2 * r2**2
                total = np.pi*(r2**2 + r1**2)
                intersect = s1 + s2
                union = total - intersect
                iou=intersect/union
                #print(i,iou)
            if iou > maxiou:
                maxiou = iou
                index_iou=i
#    if maxiou > 0:
    return np.array([-maxiou, index_iou, mind, index_dist])
    #return np.array([mind,index_dist])

@numba.jit
def iouD_all(my_x,my_y,my_d,xv,yv,dv,dxP,dyP, ddP, radius=3391,method="Benedix"):
    """Find the IOU and distances for all candidate craters.
    my_* -> my crater
    ?v -> candidate craters list/array
    dxP, dxP, ddP -> percentage error in lat/long allowed
    radius -> radius of planet
    method="Bendix" or "Lee" to choose the method

    #Returns the indices of all craters, their distances and ious
    """
    #Find the maximum iou

    indices=[]
    distances=[]
    ious=[]

    count=0
    k2d_y = radius * (2*np.pi)/360
    k2d_x = k2d_y * np.cos(my_y*np.pi/180)
                

    for i in range(len(xv)):
        iou=0
        check=False
        dx = abs(my_x)*dxP/100
        dy = abs(my_y)*dyP/100

        if method=="Benedix":
            #Check D
            if np.abs(my_d - dv[i]) < (ddP/100)*my_d:
                check=True
            if check:
                this_dx = xv[i] - my_x
                if this_dx > 180: #go the other way
                    this_dx = 360 - this_dx
                if this_dx < -180:
                    this_dx = 360 + this_dx
                
                if abs(this_dx) < dx:
                    check=True
                else:
                    check=False
            #check Y        
            if check:
                if yv[i] >= (my_y-dy) and yv[i] <=my_y + dy:
                    check=True
                else:
                    check=False
        elif method=="Lee":
            if np.abs(my_d - dv[i]) < (ddP/100)*my_d:
                check=True
            if check:
                this_dx = xv[i] - my_x
                this_dy = yv[i] - my_y
                if this_dx > 180: #go the other way
                    this_dx = 360 - this_dx
                if this_dx < -180:
                    this_dx = 360 + this_dx
                    
                #X and Y
                check = (np.abs(this_dx)*k2d_x < dxP*my_d/100) and (np.abs(this_dy)*k2d_y < dyP*my_d/100)

        if not check:
            continue
        else:
            #print(i)
            count+=1
            #calc iou!
            x1 = my_x
            y1 = my_y
            r1 = my_d/2

            x2 = xv[i]
            y2 = yv[i]
            r2 = dv[i]/2

            #great_circle
            d2r = np.pi/180
            v = np.sin(y1*d2r) * np.sin(y2*d2r) + np.cos(y1*d2r)*np.cos(y2*d2r) * np.cos((x1-x2)*d2r)
            
            if v > (1 - 1e-10) and v<(1+1e-10):
                v=1
            elif v > -(1 - 1e-10) and v<-(1+1e-10):
                v=-1
            px1 = 0
            py1 = 0
            px2 = 0
            py2 = radius *np.arccos(v)
            p1 = np.array([px1,py1])
            p2 = np.array([px2,py2])
            if r1 < r2: #swap
                _ = r2
                r2 = r1
                r1 = _
                
            d = py2
            my_distance = d
            if d > r1 + r2:
                intersect = 0
                union = np.pi*r1**2 + np.pi*r2**2
                total = np.pi*r1**2 + np.pi*r2**2
                iou=intersect/union

            elif d <= r1-r2:
                intersect = np.pi*r2**2
                union = np.pi*r1**2
                total = np.pi*r1**2 + np.pi*r2**2
                iou=intersect/union
            elif d < r1+r2:
                a = (r1**2 - r2**2 + d**2)/(2*d)
                b = (r2**2 - r1**2 + d**2)/(2*d)
                if abs(r1-a)<1e-6:
                    h=0
                else:
                    h = np.sqrt(r1**2-a**2)
                p5 = p1+(a/d)*(p2-p1)
                v12 = p2-p1

                rc = np.array([[0,1.],[-1.,0]])
                v54 = (h/d)*np.dot(rc,v12)
                p4 = p5 +v54
                ra = np.array([[0,-1.],[1,0.]])
                v53 = (h/d)*np.dot(ra,v12)
                p3 = p5 + v53

                theta1 = 2*np.arccos(a/r1)
                s1 = (theta1-np.sin(theta1))/2 * r1**2
                theta2 = 2*np.arccos(b/r2)
                s2 = (theta2-np.sin(theta2))/2 * r2**2
                total = np.pi*(r2**2 + r1**2)
                intersect = s1 + s2
                union = total - intersect
                iou=intersect/union
                #print(i,iou)
                
            ious.append(iou)
            distances.append(my_distance)
            indices.append(i)
            
    return indices,distances,ious

def iouD_topN(my_x,my_y,my_d,xv,yv,dv,dxP,dyP, ddP, radius=3391, topN=10, method="Benedix"):
    """Find the top N crater candidates for this crater.

    Calls iouD_all with almost all the arguments. 
    Then scores craters based on their IOU (overlapping) or distance (non-overlapping) and returns the topN craters (default 10)
    """
    #Find the maximum iou
    indices,distances,iou = iouD_all(my_x,my_y,my_d,xv,yv,dv,dxP,dyP,ddP,radius=radius, method=method)

    score = [-i if i>0 else d for i,d in zip(iou,distances)]
    arr = zip(score,indices,iou,distances)
    return sorted(arr,key=lambda x: x[0])[:topN]

def calculate_metrics(data,catalogs, tablename="without_duplicates",xlim=(-180,180), ylim=(-65,65),dlim=(1.5,10), include_addon=True,my_catalogs_to_compare=None):
    my_catalogs_to_compare = my_catalogs_to_compare or catalogs_to_compare
    table = []
    for m in methods:
        for k in my_catalogs_to_compare:
            row = [m,k[0],k[1]]
            name=f"/{m}/{k[0]}_{k[1]}"
            matched = data[name+f"/{tablename}"]
            truth = catalogs[k[1]]
            test = catalogs[k[0]]
            
            #all truth craters within the limits
            dfG = filter_data(truth,xlim=xlim,ylim=ylim,dlim=dlim)
            #test craters within the limits
            dfT = filter_data(test,xlim=xlim,ylim=ylim,dlim=dlim)
            #all matched within the limits
            dfM = filter_data(matched,xlim=xlim,ylim=ylim,dlim=dlim)

            tp = len(dfM)
            fn = len(dfG) - len(dfM)
            fp = len(dfT) - len(dfM)
            recall=0
            precision=0
            f1=0
            if tp+fn>0:
                recall = (tp)/(tp + fn)
            if tp+fp>0:
                precision = (tp)/(tp + fp)
            if precision+recall>0:
                f1 = 2*precision*recall/(precision+recall)
            #print(recall, precision, f1)
            row.extend([recall*100, precision*100, f1*100])
            row.extend([tp,fp,fn,len(dfT),len(dfG)])
            table.append(row)
    table_names = ["method","test","truth","recall","precision","F1","TP","FP","FN","test_count","truth_count"]
    
    return pd.DataFrame(table,columns=table_names)


def print_metric_table(data,catalogs, tablename="without_duplicates",form="latex", my_catalogs_to_compare=None,
                       include_addon=True,xlim=(-180,180),ylim=(-65,65),dlim=(1.5,10)):
    print("Reading data")
    import pickle
    my_catalogs_to_compare = my_catalogs_to_compare or catalogs_to_compare

    table = calculate_metrics(data,catalogs, tablename=tablename, my_catalogs_to_compare=my_catalogs_to_compare,
                              include_addon=include_addon,xlim=xlim,ylim=ylim,dlim=dlim)
    if form=="latex":
        with open(paths.figures/"table1.tex",'w') as handle:
            with pd.option_context('display.float_format', '{:,.1f}'.format):     
                table = table.rename(columns=dict((k, k.replace("_"," ")) for k in table.columns))
                del table["truth"]
                del table["truth count"]
                table = table.rename(columns=dict(recall="R",precision="P"))
                handle.write(table.style.format(decimal=".",thousands=",",precision=0).hide(axis="index").to_latex())
    elif form=="csv":
        with pd.option_context('display.float_format', '{:,.1f}'.format):         
            print(table.to_csv())
    elif form=="print":
        with pd.option_context('display.float_format', '{:,.1f}'.format):         
            print(table)
    elif form=="notebook":
        return table

def load_catalogs(filename):
    dest=dict()
    for k in catalogs:
        dest[k] = pd.read_hdf(filename,key=k)
    for k in human:
        dest[k] = pd.read_hdf(filename,key=k)
    return dest

def iterate_fill(input_data, data, truth, target):
    """Given a duplicated crater list, remove duplicates and find alternate candidates"""
    deduplicated = input_data.copy()
    ind = deduplicated["Robbins_Index"]
    dupes = list(set(ind[ind.duplicated()].values))
    to_delete = []

    for dupe_ri in dupes:
        v = input_data[input_data["Robbins_Index"]==dupe_ri]
        #minimize the score in this group
        score = np.where(v.iou >0, -v.iou, v["Distance (km)"])
        choose = np.argmin(score)
        my_delete = [v.iloc[i].name for i in range(len(v)) if i != choose]
        to_delete.extend(my_delete)

    #now delete them all
    deduplicated = deduplicated.drop(to_delete)
    #now see if there's a replacement
    replacements=[]
    print("deleted {}".format(len(to_delete)))
    print("remaining {}".format(len(deduplicated)))
    for missing in to_delete:
        index = int(input_data.loc[missing]["Catalog_Index"])
        for r in data[index]:
            if r[1] not in deduplicated["Robbins_Index"].values:
                rob = truth.iloc[r[1]]
                inp = target.loc[index]
                replacement = [rob.name,r[2],r[3],rob.Long,rob.Lat,rob["Diameter (km)"],inp.name,inp.Long,inp.Lat,inp["Diameter (km)"]]
                replacements.append(replacement)
                break

    replacements = pd.DataFrame(replacements, columns=["Robbins_Index","iou","Distance (km)","Long","Lat","Diameter (km)","Catalog_Index","matched_Long","matched_Lat","matched_Diameter (km)"])

    output = pd.concat([deduplicated,replacements], ignore_index=True)
    print(len(input_data), len(deduplicated), len(output))
    return output, len(input_data) != len(output)

def fill_matches(catalog_file, target_filename,truth_name,target_name):
    naive = []
    data = pickle.load(open(target_filename,'rb'))
    target = pd.read_hdf(catalog_file, target_name)
    truth = pd.read_hdf(catalog_file, truth_name)

    for index,r in enumerate(data):
        if len(r):# and r[0][0] < 0: # score < 0 -> IOU
            rob = truth.iloc[r[0][1]]
            inp = target.iloc[index]
            x=[rob.name,r[0][2],r[0][3],rob.Long,rob.Lat,rob["Diameter (km)"],inp.name,inp.Long,inp.Lat,inp["Diameter (km)"]]
            naive.append(x)
    naive=pd.DataFrame(naive, columns=["Robbins_Index","iou","Distance (km)","Long","Lat","Diameter (km)","Catalog_Index","matched_Long","matched_Lat","matched_Diameter (km)"])

    for name in ["Robbins_Index","Catalog_Index"]:
        naive[name] = naive[name].astype(int)


    #now deduplicate

    new = naive.copy()
    changed = True
    print("Length at start", len(new))
    while changed:
        new,changed = iterate_fill(new, data, truth, target)
    for name in ["Robbins_Index","Catalog_Index"]:
        new[name] = new[name].astype(int)
    return naive, new

def match_catalogs(catalog_file, methods, catalogs,
                     output_filename="processed/v2_comparison_data.h5",
                     flags=None,xlim=(-180,180),ylim=(-90,90),dlim=(0,1000), replace=None
                     ):
    replace = replace or []
    suffix = None
    if suffix is None:
        suffix = "{}_{}_{}_{}_{}_{}".format(xlim[0],xlim[1],ylim[0],ylim[1],dlim[0],dlim[1])
    output_file = Path(output_filename)
    output_file = str(output_file.parent/output_file.stem)+f"_{suffix}"+output_file.suffix
    print("xlim limited to {}".format(xlim))
    print("ylim limited to {}".format(ylim))
    print("dlim limited to {}".format(dlim))
    print("output file is {}".format(output_file))
    if not (paths.data/output_file).parent.exists():
        (paths.data/output_file).parent.mkdir(exist_ok=True,parents=True)
    if (paths.data/output_file).exists():
        output = pd.HDFStore(str(paths.data/output_file),'a')
    else:
        output = pd.HDFStore(str(paths.data/output_file),'w')

    for method in methods:
        for target_name, truth_name in catalogs:
            print(method,target_name,truth_name)
            name=f"/{method}/{target_name}_{truth_name}"
            name_with_duplicates = name + "/with_duplicates"
            name_without_duplicates = name + "/without_duplicates"
            if (name_with_duplicates in output) and (name_with_duplicates not in replace):
                print(f"NAME IN OUTPUT: {name}")
                continue
            target_filename = paths.data/f"interim/{method}_{target_name}_{truth_name}.pkl"
            #target = catalogs[target_name]
            #truth  = catalogs[truth_name]
            with_duplicates, without_duplicates = fill_matches(catalog_file, target_filename, truth_name, target_name)        
            
            output[name_with_duplicates] = with_duplicates
            output[name_without_duplicates] = without_duplicates

    output.close()



def plot_bias(data):
    """Plots the calculated bias in Benedix catalog.
    
    Raw catalogs are read in and parameters calculated to get the dx and dy pixel values,
    then the scales relative to diameter are calculated (1 of 3 values).

    subplot 1: shape of the rectangles (ratio of dy/dx) as a function of latitude is calculated as a heatmap of crater population.

    subplot 2: diameter bias as function of latitude for all the matches using Benedix method.
    """
    
    df = pd.read_csv(paths.data/"external/ess2_502-sup-0002-2019ea001005-tdata_set_si1.csv")
    df2 = pd.read_csv(paths.data/"external/ess_rescaled.csv")
    
    df["dx"] = df["Pixel value x2"] - df["Pixel value x1"]
    df["dy"] = df["Pixel value y2"] - df["Pixel value y1"]
    df["sx"] = df["diameter (km)"]/df["dx"]
    df["sy"] = df["diameter (km)"]/df["dy"]
    df["ratio"] = df["dy"]/df["dx"]
    
    fig, axs = plt.subplots(2,1,figsize=(4,6),sharex=True)
    axs[0].hist2d(df2.latitude, df2.dy/df2.dx, bins=[np.arange(-65,70,5), np.arange(0,2,.1)], cmap='Reds');
    y = np.arange(-65,65,5)
    axs[0].plot(y, np.cos(np.deg2rad(y)),color='k');
    axs[0].set_xlim(-65,65)
    axs[0].set_xticks([-60,-30,0,30,60])
    #axs[0].set_xlabel("Latitude")
    axs[0].set_ylabel("Ratio of pixel dy/dx")
    axs[0].grid()
    
    for k in plot_names:
        q=data[f"/Lee/{k}_Robbins/without_duplicates"]
        q["alat"] = np.abs(q["Lat"])
        r=[]
        for l in np.arange(-65,65):
            q2 = q.query(f"Lat > {l-1} and Lat < {l+1}")
            mm = np.array([min(a,b) for a,b in zip(q2["Diameter (km)"],q2["matched_Diameter (km)"])])
            m = np.median((100*((q2["matched_Diameter (km)"] - q2["Diameter (km)"])/mm)))
            r.append([l,m])
            if l==0:
                meq=m
        r = np.array(r).T
        if meq !=0:
            r[1] = r[1]/meq
        else:
            r[1][:]=1
        
        axs[1].plot(r[0],r[1],label=labels.get(k,k),color=colors[k],alpha=1 if k.startswith("Be") else 0.5)
    for k in plot_names:
        q=data[f"/Benedix/{k}_Robbins/without_duplicates"]
        q["alat"] = np.abs(q["Lat"])
        r=[]
        for l in np.arange(-65,65):
            q2 = q.query(f"Lat > {l-1} and Lat < {l+1}")
            m = np.median((100*((q2["matched_Diameter (km)"] - q2["Diameter (km)"])/q2["Diameter (km)"])))
            r.append([l,m])
            if l==0:
                meq=m
        r = np.array(r).T
        if meq !=0:
            r[1] = r[1]/meq
        else:
            r[1][:]=1
        
        axs[1].plot(r[0],r[1],ls='--',color=colors[k],alpha=1 if k.startswith("Be") else 0.5)
    axs[1].legend(ncol=2)
    plt.setp(axs[1], xlabel="Latitude",ylabel="scaled diameter error", ylim=(0,5))
    axs[1].grid()


def calc_distance(name,entry, override_method=None):
    #only data that matches something
    data = entry.copy()
    delta = dict()
    
    delta["Long"] = data["Long"] - data["matched_Long"]
    delta["Lat"] = data["Lat"] - data["matched_Lat"]    
    
    delta["Diameter"] = 100*(data["matched_Diameter (km)"] - data["Diameter (km)"])/np.abs(data["matched_Diameter (km)"])
    
    override_method=override_method or name.startswith("/Lee")
    
    if override_method:
        #percentage relative to size
        k2d_y = config.R_planet * 2*np.pi/360.
        k2d_x = config.R_planet * 2*np.pi *np.cos(np.deg2rad(data["Lat"]))/360.
        delta["Lat"] = 100*k2d_y*delta["Lat"]/(1e-5 + data["Diameter (km)"])
        delta["Long"] = 100*k2d_x*delta["Long"]/(1e-5 + data["Diameter (km)"])
    else:
        #absolute percentage
        delta["Long"] = 100*delta["Long"]/(1e-5 + np.abs(data["Long"]))
        delta["Lat"] = 100*delta["Lat"]/(1e-5 + np.abs(data["Lat"]))
        
    return delta

    
def plot_distance(catalogs, keys,ranges=None, override_method=None,
                 xlim=(-180,180), ylim=(-65,65), dlim=(1.5,10)):
    """Plots the spatial errors for each catalog in 3 subplots

    catalogs: The HDF catalog file
    keys : list of keys to plot
    ranges(=None) : dictionary of "Lat","Long","Diameter" values to use in the aggregation.
    xlim,ylim,dlim : filter limits on the catalog
    
    Returns : 
       delta : The differences in a dictionary
    """


    ranges = ranges or {"Long":np.linspace(-30,30,101),
                  "Lat":np.linspace(-30,30,101),
                  "Diameter":np.linspace(-30,30,101),
             }
    
    fig, axs = plt.subplots(1,3,figsize=(12,3), sharey=True)

    for name in keys:
        entry = filter_data(catalogs[name],xlim=xlim,ylim=ylim,dlim=dlim)
        delta = calc_distance(name,entry,override_method=override_method)
        

        for k,a in zip(ranges.keys(),axs):
            myname = name.split("/")[2].split("_")[0]
            delta[k].hist(bins=ranges[k],ax=a,color=colors[myname],label=labels.get(myname,myname),histtype='step')

            a.set_yscale('log')
            a.legend(frameon=False, ncol=2, loc=(0.10,0.7))
            kname = labels.get(k,k)
            a.set_xlabel(f"{kname} error (%)")
            a.set_xlim(ranges[k].min(), ranges[k].max())
        axs[0].set_ylabel("Number of matches")
    return delta


def plot_iou_density(data, keys, tablename="without_duplicates",iou_range=None):
    """Plots the IOU density plot for each catalog with both methods.

    Plots IOU in a split plot with two limits of (0-10),(10,100) to highlight the lower 
    density values and the high peaks at IOU=0,1.

    data: HDF data
    keys : list of keys to plot
    tablename(="without_duplicates") : which table to use, "with_duplicates" or "without_duplicates"
    iou_range(=None) : if present, the values of IOU to use in the histogram.


    """
    if iou_range is None:
        iou_range = np.arange(0.0,1.01,0.01)
    
    fig,(ax2,ax1) = plt.subplots(2,1,figsize=(4,4), height_ratios=[1,3])
    for method,ls  in zip(["Lee","Benedix"],["-","--"]):
        for lr, name in enumerate(keys):
            a=data[f"/{method}/{name}/{tablename}"]
            myname = name.split("_")[0]
            for ax in [ax1,ax2]:
                a["iou"].hist(bins=iou_range,ax=ax, color=colors[myname], label=labels.get(myname,myname), histtype='step',density=True,ls=ls)
                if method=="Benedix":
                    yloc = sum(a["iou"]==0)*100/len(a["iou"])
                    ax.arrow(0.1*(-1)**lr,yloc,(-1)**(lr+1)*0.08,0, length_includes_head=True, head_length=0.01, color=colors[myname])
                    print(yloc, (-1)**lr)
       # ax.set_yscale('log')
        if method=="Lee":
            ax2.legend(frameon=False, ncol=1, fontsize=9, loc=(0.5,-0.1))
        ax1.set_xlabel("IOU")
        ax1.set_ylabel("Density")
        ax1.set_ylim(0,6.5)
        ax2.set_ylim(10,20)
        ax2.set_yticks([10,15,20])
        
        ax2.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        ax2.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax1.xaxis.tick_bottom()

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
        ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        r = 1/3
        kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
        ax1.plot((-d, +d), (1 - d*r, 1 + d*r), **kwargs)  # bottom-left diagonal
        ax1.plot((1 - d, 1 + d), (1 - d*r, 1 + d*r), **kwargs)  # bottom-right diagonal

def plot_metric_vs_Lat(data, catalogs, keys,ranges=None,ls="-",alpha=1, axs=None,precision=False,**limits):
    """Plots metric (recall and precision) against latitude.

    data : the data dictionary from the HDF file
    catalogs : the raw catalogs to get the ground truth dataset
    keys : list of catalogs to plots
    ranges : dictionary of Long,Lat,"Diameter (km)" keys and values that are bins used in the aggregation
    alpha(=1) : opacity used in the plotting
    axs(=None) : If present contains 3 axes to plot additional data on. If not present, it will be created.
    precision(=False) : if True calculates and plots the precision instead of recall
    **limits : filter limits if present

    Returns :  
      axs : 3 axes used for plotting.
"""

    ranges = ranges or {"Long":np.arange(-180,180+5,5),
                        "Lat":np.arange(-65,65+5,5),
                        "Diameter (km)": np.logspace(0,2,41)
             }
    plot_limits = {"Long": [-180,180], "Lat":[-65,65],"Diameter (km)": [1,10]}
    legend=False
    if axs is None:
        fig, axs = plt.subplots(1,3,figsize=(12,3), sharey=True)
        legend=True
    for a,(range_key, range_values) in zip(axs,ranges.items()):
        for name in keys:

            entry = filter_data(data[name],**limits)
            if precision:
                tname = name.split("/")[2].split("_")[0]
            else:
                tname = name.split("/")[2].split("_")[1]

            mytruth = catalogs[tname].copy()
            mytruth["TPR"] = False
            if precision:
                mytruth.loc[entry["Catalog_Index"],"TPR"] = True
            else:
                mytruth.loc[entry["Robbins_Index"],"TPR"] = True

            mytruth = filter_data(mytruth,**limits)

            grps = mytruth.groupby(pd.cut(mytruth[range_key],range_values))
            mn = grps.mean()
            st = grps.std()
            myname = name.split("/")[2].split("_")[0]
            a.errorbar(mn[range_key],mn["TPR"],label=labels.get(myname,myname),ls=ls,color=colors[myname],alpha=alpha)
            
            a.set_xlabel(labels.get(range_key, range_key))
            a.set_xlim(plot_limits[range_key])
    axs[0].set_ylabel("Recall or Precision")

    if legend:
        axs[2].legend(frameon=False,ncol=2)
    
    return axs



def naive_csfd(catalogs,dlim=(1.5,100),**limits):
    """Plots a relatively simple CSFD from the catalogs.
    filters in space, bins into 20 log bins per decade.
    """
    
    plt.figure(figsize=(3,6))
    plt.gca().set_aspect('equal')
    for name in catalogs.keys():
        cat = filter_data(catalogs[name],dlim=dlim,**limits)
        L,H = np.log10(dlim[0]),np.log10(dlim[1])
        N = (H-L)*20
        D = (H-L)/N
        N = round(N+0.5)
        H = (L+N*D)
        c,m = (cat.groupby(pd.cut(cat["Diameter (km)"],np.logspace(L,H,N))).count()["Diameter (km)"],
               cat.groupby(pd.cut(cat["Diameter (km)"],np.logspace(L,H,N))).mean()["Diameter (km)"])
        plt.plot(m,c,marker='|',label=labels.get(name,name),color=colors[name],alpha=0.76)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(dlim)
    plt.legend(ncol=1, frameon=True)
    plt.grid()
    plt.xlabel("Diameter (km)")
    plt.ylabel("Count")
