##location and distance calculation

##TODO: quantize distance so that some subrange local_factor are shared, so computation can speed up using look up table


##localization function
def local_factor(dist, ROI, local_type='GC'):
    ## dist: input distance, ndarray
    ## ROI: radius of influence, distance beyond which loc=0
    ## returns the localization factor loc
    loc = np.zeros(dist.shape)
    if ROI>0:
        if local_type == 'GC': ##Gaspari-Cohn localization function
            r = dist / (ROI / 2)
            loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
            ind1 = np.where(dist<ROI/2)
            loc[ind1] = loc1[ind1]
            r[np.where(r==0)] = 1e-10
            loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
            ind2 = np.where(np.logical_and(dist>=ROI/2, dist<ROI))
            loc[ind2] = loc2[ind2]
        else:
            raise ValueError('unknown localization function type: '+local_type)
    else:
        loc = np.ones(dist.shape)
    return loc
