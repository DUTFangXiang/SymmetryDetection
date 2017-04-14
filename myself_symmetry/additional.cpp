
long int rint(double x)
{
    if ( int(x) == x )
	    return (long)( x );
	else
	    return (long)(x + 1);

  }
