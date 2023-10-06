# -*- coding: UTF-8 -*-
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import re
import time
import os 

def up_down(value):
    return 'Increase' if value>0 else 'Decrease'

def percent_format(v,d=2):#percent_number_format
    if d<0:
        raise ValueError("percent_format(v,d),d should be >=0")
    try:
        return str(int(round(v*100,d))) + "%" if d == 0 else str(round(v*100,d)) + "%"
    except Exception as e:
        raise ValueError(str(v) + " is not an float number!")

def node_text_format(text):
    text = text.replace("not not","")
    return text

def get_info_for_tree(value_type,df):
        if value_type == 'quantity':
            this_sum = df['this'].sum()
            base_sum = df['base'].sum()
            diff_total = this_sum - base_sum
            root_node_text = "base:" + str(int(base_sum)) +"\n" + "this:" + str(int(this_sum)) + "\n" 
            root_node_text += up_down(diff_total) +":"+ str(abs(int(diff_total))) 
            root_node_text += "(" +str(int(base_sum))+"->"+str(int(this_sum))
            root_node_text += ","+percent_format(this_sum/base_sum-1)+")"
        elif value_type == 'rate':
            this_sum = df["n_this"].sum()/df["d_this"].sum()
            base_sum = df["n_base"].sum()/df["d_base"].sum()
            diff_total = this_sum - base_sum
            root_node_text = "base:" + percent_format(base_sum) +"\n" + "this:" + percent_format(this_sum) + "\n"
            root_node_text +=  up_down(diff_total) +":"+ percent_format(abs(diff_total)) 
            root_node_text += "(" + percent_format(base_sum) + "->" + percent_format(this_sum) 
            root_node_text += "," + percent_format(this_sum/base_sum-1)+")"
        return diff_total,root_node_text

def byte_len(value):
    length = len(value)
    utf8_length = len(value.encode('utf-8'))
    length = (utf8_length - length)/2*5/3 + length-(utf8_length - length)/2
    return int(length)

def eval_tree_leafs(tree):
    leafs = []
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            leafs += eval_tree_leafs(secondDict[key])
        else:
            leafs.append(secondDict[key])
    return leafs

def get_node_direction(text):
    f = None
    infos = text.split("\n")
    for info in infos:
        t = re.sub("\(.*\)","",info.split(":")[-1])
        if 'Increase' in info:
            if t.endswith("%"):
                f = float(t.replace("%",""))/100
            else:
                f = int(float(t))
        if 'Decrease' in info:
            if t.endswith("%"):
                f = -float(t.replace("%",""))/100
            else:
                f = -int(float(t))
    if f == None:
        return 0
    else:
        return f

def color_trans(value):#useless for now
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        
def node_color(diff_value,diff_total):#useless for now
    if (diff_value/diff_total<1.0) & (diff_value/diff_total>0.0):
        if diff_value<0:
            return color_trans((int(55*diff_value/diff_total+200),0,0))
        else:
            return color_trans((0,int(55*diff_value/diff_total+200),0))
    else:
        if diff_value<0:
            return color_trans((255,0,0))
        else:
            return color_trans((0,255,0))

def node_alpha(nodeTxt):
    try:
        if nodeTxt == '':
            x = 0.9
        else:
            t = nodeTxt.split(":")[-1]
            t = float(t.replace("%",""))/100
            if (t < 1.0) & (t > 0.0):
                x = t
            elif (t < 0.0) & (abs(t)<0.3):
                x = 0.1
            else:
                x = 0.9
        return x*0.8
    except Exception as e:
        return 0.1*0.8

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args,color='black' )
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

arrow_args = dict(arrowstyle="<|-",color='gray',alpha=0.5)
def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]    #te text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    diff_value = get_node_direction(firstStr)
    if diff_value>0:
        plotNode(firstStr, cntrPt, parentPt, dict(edgecolor='white',boxstyle="sawtooth", fc="green",alpha=node_alpha(nodeTxt)))
    else:
        plotNode(firstStr, cntrPt, parentPt, dict(edgecolor='white',boxstyle="sawtooth", fc="red",alpha=node_alpha(nodeTxt)))

    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            diff_value = get_node_direction(secondDict[key])
            if diff_value>0:
                plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, dict(edgecolor='white',boxstyle="round4", fc="green",alpha=node_alpha(key)))
            else:
                plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, dict(edgecolor='white',boxstyle="round4", fc="red",alpha=node_alpha(key)))
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def plot_node_width(tree):
    max_byte_len = 0
    for lf in eval_tree_leafs(tree):
        for l in lf.split("\n"):
            if byte_len(l)>max_byte_len:
                max_byte_len = byte_len(l)
    return max(int(max_byte_len/8),1)

def createPlot(tree_for_create):
    fig = plt.figure(figsize=(getNumLeafs(tree_for_create)*plot_node_width(tree_for_create),getTreeDepth(tree_for_create)*3),facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    plotTree.totalW = float(getNumLeafs(tree_for_create))
    plotTree.totalD = float(getTreeDepth(tree_for_create))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(tree_for_create, (0.5,1.0), '')
    plt.show()
    plt.tight_layout()
    time.sleep(1)
    png_name = "dim_attribution_" + str(int(time.time())) + ".png"
    fig.savefig(png_name)
    png_path = os.path.abspath(os.path.join(os.getcwd(), ".")) + "/" + png_name
    plt.close()
    # print("png_path:",png_path)
    return png_path


class DimtributorTree:
    def __init__(self,df_initial,value_type,group,dims,y,d='',n='',max_tree_depth=3,min_root_weight=0.3):
        self.df_initial = df_initial
        self.value_type = value_type
        self.group = group
        self.dims = dims
        self.y = y
        self.d = d#denominator
        self.n = n#numerator
        self.max_tree_depth = max_tree_depth #MAX_TREE_DEPTH
        self.min_root_weight = min_root_weight #MIN_ROOT_WEIGHT
        
    def df_check(self):
        if not isinstance(self.df_initial,pd.DataFrame) :#make sure df_initial is a pandas dataframe
            raise ValueError('in def df_check:not a pandas dataframe')
        if self.value_type not in ['quantity','rate']:#check for value_type
            raise ValueError('in def df_check:value_type could only be quantity or rate')
        if self.group not in self.df_initial:#check for group if in df_initial
            raise ValueError('in def df_check:group not in df_initial')
        if set(self.df_initial[self.group].values) != {'base','this'}:#check for group
            raise ValueError('in def df_check:group should include and only include base and this')
        if not isinstance(self.dims,list):#check for dims
            raise ValueError('in def df_check:dims should be a list')
        if not np.all([i in self.df_initial.columns for i in self.dims]):#check for dims
            raise ValueError('in def df_check:at least one item in dims not in df_initial')
        if self.y not in self.df_initial:#check for y if in df_initial
            raise ValueError('in def df_check:y not in df_initial')
        if self.value_type=='rate':
            if self.d == '' or self.n == '':
                raise ValueError('in def df_check:d and n could not be empty if value_type=rate')
            if self.d not in self.df_initial or self.n not in self.df_initial:
                raise ValueError('in def df_check:d and n not in df_initial')
    def df_prepare(self):
        if self.value_type == 'quantity':
            self.df_initial[self.y] = self.df_initial[self.y].astype(float)
            self.df_initial = self.df_initial.rename(columns={self.y:"y"})
            self.df_initial = pd.pivot_table(self.df_initial,index=self.dims,columns=self.group,values='y',aggfunc=[np.sum])
        if self.value_type == 'rate':
            self.df_initial[self.d] = self.df_initial[self.d].astype(float)
            self.df_initial[self.n] = self.df_initial[self.n].astype(float)
            self.df_initial = self.df_initial.rename(columns={self.d:"d",self.n:"n"})
            self.df_initial = pd.pivot_table(self.df_initial,index=self.dims,columns=self.group,values=["d","n"],aggfunc=[np.sum])
        self.df_initial = self.df_initial.fillna(0)
        self.df_initial.columns = ["_".join(i)[4:] for i in self.df_initial.columns]
        self.df_initial = self.df_initial.reset_index(drop=False)

    def choose_best_split_js(self,df):    
        f = {"dim":''
                    ,"value":''
                    ,"percent_parent_node":0.0
                    ,"percent_root_node":0.0
                    ,"dim_js":0.0
                    ,"value_js":1.0}
        dim_js = 0.0
        value_js = 1.0
        best_split_dimension = ''
        best_split_value = 0.0
        percent_parent_node = 0.0
        percent_root_node = 0.0
        
        for dim in self.dims:#best_split_dimension
            if df[dim].drop_duplicates().shape[0]==1:
                continue
            try: 
                js = self.cal_dim_js(df,dim,exclude_value='')
            except Exception as e:
                #print('cal_dim_js,error for dim:',dim,",exclude_value:",dim_value,",error_info:",str(e))
                continue
            if js > dim_js:
                dim_js = js
                best_split_dimension = dim
        if best_split_dimension == '':
            return {"dim":''
                    ,"value":''
                    ,"percent_parent_node":0.0
                    ,"percent_root_node":0.0
                    ,"dim_js":0.0
                    ,"value_js":1.0}
        for dim_value in df[best_split_dimension].drop_duplicates().values:#in best_split_dimension choose best_split_value
            try: 
                js = self.cal_dim_js(df,best_split_dimension,exclude_value=dim_value)
            except Exception as e:
                #print('cal_dim_js,error for dim:',dim,",exclude_value:",dim_value,",error_info:",str(e))
                continue
            if js < value_js:
                value_js = js
                best_split_value = dim_value
        #print(best_split_dimension,best_split_value)
        df_tmp = df[df[best_split_dimension]==best_split_value]
        if self.value_type == 'quantity':
            diff_tmp = get_info_for_tree(self.value_type,df_tmp)[0]
            diff_parent = get_info_for_tree(self.value_type,df)[0]
            diff_root = get_info_for_tree(self.value_type,self.df_initial)[0]
            percent_parent_node = diff_tmp/diff_parent
            percent_root_node = diff_tmp/diff_root
        elif self.value_type == 'rate':
            if df_tmp['d_this'].sum() == 0 :
                return f 
            tmp_this = df_tmp['n_this'].sum()/df_tmp['d_this'].sum()*df_tmp['d_this'].sum()/df['d_this'].sum()
            tmp_base = df_tmp['n_base'].sum()/df_tmp['d_base'].sum()*df_tmp['d_base'].sum()/df['d_base'].sum()
            diff_tmp = tmp_this - tmp_base
            diff_parent = get_info_for_tree(self.value_type,df)[0]
            percent_parent_node = diff_tmp/diff_parent

            tmp_this = df_tmp['n_this'].sum()/df_tmp['d_this'].sum()*df_tmp['d_this'].sum()/self.df_initial['d_this'].sum()
            tmp_base = df_tmp['n_base'].sum()/df_tmp['d_base'].sum()*df_tmp['d_base'].sum()/self.df_initial['d_base'].sum()
            diff_tmp = tmp_this - tmp_base
            diff_root = get_info_for_tree(self.value_type,self.df_initial)[0]
            percent_root_node = diff_tmp/diff_root

        if abs(percent_root_node)>self.min_root_weight: #MIN_ROOT_WEIGHT:
            f = {"dim":best_split_dimension
                    ,"value":best_split_value
                    ,"percent_parent_node":percent_parent_node
                    ,"percent_root_node":percent_root_node
                    ,"dim_js":dim_js
                    ,"value_js":value_js}
        return f 

    def df_split(self,df,dimension,value=1):
        df1 = df.loc[df[dimension]==value,]
        df2 = df.loc[df[dimension]!=value,]
        return df1,df2
    
    def cal_dim_js(self,df,dim,exclude_value=''):
        df = df[df[dim]!=exclude_value]
        if self.value_type == 'quantity':
            df_tmp = df.groupby(dim).agg({"this":"sum","base":"sum",dim:"count"})
            if df_tmp.shape[0] <= 1:
                return 1.0
            df_tmp['this_should'] = df_tmp["base"]*(df_tmp["this"].sum()/df_tmp["base"].sum())
            js = distance.jensenshannon(df_tmp['this'],df_tmp['this_should'])
        elif self.value_type == 'rate':
            agg_hash = {"d_base":"sum","d_this":"sum","n_base":"sum","n_this":"sum",dim:"count"}
            df_tmp = df.groupby(dim).agg(agg_hash)
            if df_tmp.shape[0] <= 1:
                return 1.0
            df_tmp['this_rate_row'] = df_tmp["n_this"]/df_tmp["d_this"].sum() #df_tmp['this_rate_d_share']*df_tmp['this_rate']
            df_tmp['base_rate_row'] = df_tmp["n_base"]/df_tmp["d_base"].sum() #df_tmp['base_rate_d_share']*df_tmp['base_rate']
            js = distance.jensenshannon(df_tmp['this_rate_row'],df_tmp['base_rate_row'])
        return js

    def recursion_createtree(self,df=pd.DataFrame(),node_text_start=[],depth=0):
        if df.shape[0] == 0:
            df = self.df_initial
        diff_root = get_info_for_tree(self.value_type,self.df_initial)[0]
        if self.value_type=='quantity':
            #df['diff'] = df['this'] - df['base']
            df = df.assign(diff=df['this'] - df['base'])
            this_sum = df['this'].sum()
            base_sum = df['base'].sum()
            percent_root_node = df['diff'].sum()/diff_root#分叉前占根节点的比例
            node_text = "\n".join(node_text_start)+ "\n" if len(node_text_start)>0 else ''
            node_text += up_down(this_sum-base_sum) +":" 
            node_text += str(abs(int(this_sum-base_sum))) 
            node_text += "\n" + str(int(base_sum)) + "->" + str(int(this_sum))+"," + percent_format(this_sum/base_sum-1) 
            node_text += "\n account for root:" + percent_format(percent_root_node)
            node_text = node_text_format(node_text)
        elif self.value_type == 'rate':
            this_rate = df["n_this"].sum()/df["d_this"].sum()
            base_rate = df["n_base"].sum()/df["d_base"].sum()
            this_rate_d_share = df["d_this"].sum()/self.df_initial["d_this"].sum() 
            base_rate_d_share = df["d_base"].sum()/self.df_initial["d_base"].sum() 
            diff_this_node = this_rate_d_share*this_rate - base_rate_d_share*base_rate
            percent_root_node = diff_this_node/diff_root#分叉前占根节点的比例
            node_text = "\n".join(node_text_start)+ "\n" if len(node_text_start)>0 else ''
            node_text += up_down(diff_this_node)+":" 
            node_text += percent_format(abs(diff_this_node)) 
            node_text += "\n percentage:" + percent_format(base_rate_d_share) + "->" +percent_format(this_rate_d_share)
            node_text += "=" +  percent_format(this_rate_d_share-base_rate_d_share)
            node_text += "\n rate:" + percent_format(base_rate) + "->" +percent_format(this_rate)
            node_text += "=" +  percent_format(this_rate-base_rate)
            node_text += "\n account for root:" + percent_format(percent_root_node) 
            node_text = node_text_format(node_text) 

        if df.shape[1] == 1:
            return node_text
        if depth > self.max_tree_depth: #MAX_TREE_DEPTH:
            return node_text

        mytree = {node_text:{}}
        best_split = self.choose_best_split_js(df)
        if best_split["dim"] == '':
            return node_text
        df_l,df_r = self.df_split(df,best_split["dim"],best_split["value"])

        node_text_start_l = best_split['dim']+ ":" + str(best_split['value'])
        split_text_l =  node_text_start_l + "\n account for parent:" + percent_format(best_split["percent_parent_node"])
        split_text_l = node_text_format(split_text_l) 
        mytree[node_text][split_text_l] = self.recursion_createtree(df_l,node_text_start + [ node_text_start_l ],depth+1)

        node_text_start_r = best_split['dim']+ ":not " + str(best_split['value'])
        split_text_r = node_text_start_r + "\n account for parent:" + percent_format(1-best_split["percent_parent_node"])
        split_text_r = node_text_format(split_text_r) 
        mytree[node_text][split_text_r] = self.recursion_createtree(df_r,node_text_start + [ node_text_start_r ],depth+1)
        return mytree

    # def createPlot(self):
    #     fig = plt.figure(figsize=(getNumLeafs(self.outtree)*plot_node_width(self.outtree),getTreeDepth(self.outtree)*3),facecolor='white')
    #     fig.clf()
    #     axprops = dict(xticks=[], yticks=[])
    #     self.createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #     plotTree.totalW = float(getNumLeafs(self.outtree))
    #     plotTree.totalD = float(getTreeDepth(self.outtree))
    #     plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    #     plotTree(self.outtree, (0.5,1.0), '')
    #     plt.show()
    #     plt.tight_layout()
    #     # return fig
    #     time.sleep(1)
    #     png_name = "dim_attribution_" + str(int(time.time())) + ".png"
    #     fig.savefig(png_name)
    #     png_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + png_name
    #     plt.close()
    #     print("png_path:",png_path)
    #     self.png_path = png_path

    def createtree(self):
        self.df_check()
        self.df_prepare()
        self.outtree = self.recursion_createtree()
        self.png_path = createPlot(self.outtree)


    