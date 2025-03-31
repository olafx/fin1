#pragma once

#include <algorithm>

namespace fin1
{
namespace options
{
namespace European
{

struct Call
{ double K; // strike

  double price
  ( double ST
  ) const
  { return std::max(0., ST-K);
  }

  double price
  ( const double *S, size_t size_S
  ) const
  { return price(S[size_S-1]);
  }
};

struct Put
{ double K; // strike

  double price
  ( double ST
  ) const
  { return std::max(0., K-ST);
  }

  double price
  ( const double *S, size_t size_S
  ) const
  { return price(S[size_S-1]);
  }
};

namespace barrier
{

bool ever_more
( double B, const double *S, size_t size_S
)
{ for (size_t i = 0; i < size_S; i++)
    if (S[i] >= B)
      return true;
  return false;
}

bool ever_less
( double B, const double *S, size_t size_S
)
{ for (size_t i = 0; i < size_S; i++)
    if (S[i] <= B)
      return true;
  return false;
}

struct UpInCall
{ double K; // strike
  double B; // barrier

  double price
  ( const double *S, size_t size_S
  ) const
  { return ever_more(B, S, size_S) ? Call(K).price(S, size_S) : 0;
  }
};

struct UpInPut
{ double K; // strike
  double B; // barrier

  double price
  ( const double *S, size_t size_S
  ) const
  { return ever_more(B, S, size_S) ? Put(K).price(S, size_S) : 0;
  }
};

struct DownInCall
{ double K; // strike
  double B; // barrier

  double price
  ( const double *S, size_t size_S
  ) const
  { return ever_less(B, S, size_S) ? Call(K).price(S, size_S) : 0;
  }
};

struct DownInPut
{ double K; // strike
  double B; // barrier

  double price
  ( const double *S, size_t size_S
  ) const
  { return ever_less(B, S, size_S) ? Put(K).price(S, size_S) : 0;
  }
};

struct UpOutCall
{ double K; // strike
  double B; // barrier

  double price
  ( const double *S, size_t size_S
  ) const
  { return ever_more(B, S, size_S) ? 0 : Call(K).price(S, size_S);
  }
};

struct UpOutPut
{ double K; // strike
  double B; // barrier

  double price
  ( const double *S, size_t size_S
  ) const
  { return ever_more(B, S, size_S) ? 0 : Put(K).price(S, size_S);
  }
};

struct DownOutCall
{ double K; // strike
  double B; // barrier

  double price
  ( const double *S, size_t size_S
  ) const
  { return ever_less(B, S, size_S) ? 0 : Call(K).price(S, size_S);
  }
};

struct DownOutPut
{ double K; // strike
  double B; // barrier

  double price
  ( const double *S, size_t size_S
  ) const
  { return ever_less(B, S, size_S) ? 0 : Put(K).price(S, size_S);
  }
};

} // barrier
} // European
} // options
} // fin1
