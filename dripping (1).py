#!/usr/bin/env python
# coding: utf-8

# In[6]:


def my_function(x,y):
    math = x**2 + 38*y
    return math
print(my_function(5,3))
    


# In[11]:


def help(funct,o):
    result = funct +o 
    return result
help(my_function(1,2),9)
    
    
    


# In[18]:


def dripping(funct,z,g):
    p = (funct * z **2)/2
    return p


# In[19]:


dripping(help(my_function(1,2),3),1,2)


# In[29]:


list = [x**2 for x in range(2,10)]
def new_function(list,funct):
    return funct(list)
print(new_function(list,max))
print(new_function(list,min))
print(new_function(list,len))


# In[26]:





# In[ ]:




